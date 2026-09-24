"""The Model Training tab had a safe path it never took, and a guard that never guarded.

D15 gave `train_model` a real `--dry-run`: the flag now skips the JIT enrichment
pre-flight, so validating the training SQL cannot quietly spend money on Places
and Gemini calls. That fix reached the CLI only. The tab called `train_model`
without `dry_run` at all, so the one button on it went straight to
`CREATE OR REPLACE MODEL` -- there was no way to ask "is this SQL valid?" from
the UI (D-28).

Two more things were wrong on the same fifteen lines. `training_lock` was
initialised `False`, passed to `disabled=`, and never assigned `True` anywhere
in the repo, so the double-click guard was decoration. And `run_async=True`
returns a job id that the UI printed once and never mentioned again: a training
run that failed ten minutes later looked exactly like one that succeeded.
"""
import unittest
from unittest.mock import patch


class FakeSessionState(dict):
    """`st.session_state` for the subset of its API this tab uses.

    Streamlit's real object supports attribute access too; the tab deliberately
    sticks to `.get`/`[...]` so this stays a plain dict.
    """


class ModelTrainingTabCase(unittest.TestCase):
    """Render the tab against a mocked Streamlit and a mocked training module."""

    def _render(self, pressed=(), job_status=None, status_error=None,
                session=None, train_result="job-123", train_error=None):
        """Render once.

        `pressed` is the set of button keys the user clicked on this rerun;
        `job_status` is what `training_job_status` reports for a tracked job.
        Returns (mock_st, mock_train_model, session_state).
        """
        from app.ui import st_app

        state = FakeSessionState(session or {})
        buttons = {}

        def fake_button(label, **kwargs):
            key = kwargs.get("key")
            buttons[key] = kwargs
            return key in pressed

        with patch.object(st_app, "st") as mock_st, \
             patch("scripts.train_bqml_model.train_model") as mock_train, \
             patch("scripts.train_bqml_model.training_job_status") as mock_status:
            mock_st.session_state = state
            mock_st.button.side_effect = fake_button
            if train_error is not None:
                mock_train.side_effect = train_error
            else:
                mock_train.return_value = train_result
            if status_error is not None:
                mock_status.side_effect = status_error
            else:
                mock_status.return_value = job_status

            st_app.render_model_training_tab("p", "d", "t")

        mock_st.buttons = buttons
        mock_st.status_lookup = mock_status
        return mock_st, mock_train, state

    @staticmethod
    def _messages(mock_st, kind):
        return [c.args[0] for c in getattr(mock_st, kind).call_args_list if c.args]


class TestTheDryRunPathIsReachableFromTheUi(ModelTrainingTabCase):

    def test_the_tab_offers_a_dry_run_control(self):
        st, _, _ = self._render()

        self.assertIn("btn_train_dry_run", st.buttons,
                      "the tab has no dry-run control, so the safe path D15 added "
                      "is still CLI-only")

    def test_the_dry_run_control_asks_for_a_dry_run(self):
        _, train, _ = self._render(pressed={"btn_train_dry_run"},
                                   train_result=1234567)

        train.assert_called_once()
        self.assertIs(True, train.call_args.kwargs["dry_run"])

    def test_a_dry_run_starts_no_job(self):
        _, _, state = self._render(pressed={"btn_train_dry_run"},
                                   train_result=1234567)

        self.assertIsNone(state.get("training_job_id"),
                          "a dry run left a job id behind, so it ran something")

    def test_a_dry_run_reports_what_the_query_would_scan(self):
        st, _, _ = self._render(pressed={"btn_train_dry_run"},
                                train_result=1234567)

        self.assertTrue(any("1,234,567" in m or "1234567" in m
                            for m in self._messages(st, "success")),
                        f"byte estimate not surfaced: {self._messages(st, 'success')}")

    def test_invalid_sql_is_reported_rather_than_swallowed(self):
        st, _, _ = self._render(pressed={"btn_train_dry_run"},
                                train_error=RuntimeError("Unrecognized name: pillar_typo"))

        self.assertTrue(any("pillar_typo" in m for m in self._messages(st, "error")))

    def test_the_dry_run_control_is_never_locked(self):
        """It costs nothing and answers "what would this do?" -- the question
        someone asks precisely when a job is already running."""
        st, _, _ = self._render(session={"training_job_id": "job-999"},
                                job_status={"state": "RUNNING", "error": None})

        self.assertFalse(st.buttons["btn_train_dry_run"].get("disabled", False))


class TestTheRealButtonStillTrains(ModelTrainingTabCase):

    def test_pressing_train_starts_an_async_job(self):
        _, train, _ = self._render(pressed={"btn_train_model_unified"})

        self.assertIs(False, train.call_args.kwargs["dry_run"])
        self.assertIs(True, train.call_args.kwargs["run_async"])

    def test_a_job_that_will_not_start_leaves_nothing_tracked(self):
        """Otherwise the next rerun polls an id that was never issued, and the
        button locks itself shut against a job that does not exist."""
        st, _, state = self._render(pressed={"btn_train_model_unified"},
                                    train_error=RuntimeError("403 accessDenied"))

        self.assertTrue(any("403" in m for m in self._messages(st, "error")))
        self.assertIsNone(state.get("training_job_id"))

    def test_a_started_job_is_remembered(self):
        _, _, state = self._render(pressed={"btn_train_model_unified"},
                                   train_result="job-123")

        self.assertEqual("job-123", state.get("training_job_id"),
                         "the job id was printed and dropped, so nothing can "
                         "report the outcome later")


class TestTheLockActuallyEngages(ModelTrainingTabCase):
    """Mutation check: the guard has to be driven by something that changes."""

    def test_a_running_job_disables_the_train_button(self):
        st, _, _ = self._render(session={"training_job_id": "job-999"},
                                job_status={"state": "RUNNING", "error": None})

        self.assertTrue(st.buttons["btn_train_model_unified"]["disabled"])

    def test_with_no_job_the_train_button_is_live(self):
        st, _, _ = self._render()

        self.assertFalse(st.buttons["btn_train_model_unified"]["disabled"])

    def test_a_finished_job_releases_the_lock(self):
        st, _, state = self._render(session={"training_job_id": "job-999"},
                                    job_status={"state": "DONE", "error": None})

        self.assertFalse(st.buttons["btn_train_model_unified"]["disabled"])
        self.assertIsNone(state.get("training_job_id"),
                          "a finished job is still being polled")

    def test_an_unreadable_job_releases_the_lock(self):
        """A job id that cannot be looked up must not wedge the button shut."""
        st, _, state = self._render(session={"training_job_id": "job-999"},
                                    status_error=RuntimeError("Not found: Job job-999"))

        self.assertFalse(st.buttons["btn_train_model_unified"]["disabled"])
        self.assertIsNone(state.get("training_job_id"))
        self.assertTrue(self._messages(st, "warning"))


class TestTheOutcomeIsReported(ModelTrainingTabCase):

    def test_a_running_job_says_so(self):
        st, _, _ = self._render(session={"training_job_id": "job-999"},
                                job_status={"state": "RUNNING", "error": None})

        self.assertTrue(any("job-999" in m for m in self._messages(st, "info")))

    def test_a_successful_job_is_announced(self):
        st, _, _ = self._render(session={"training_job_id": "job-999"},
                                job_status={"state": "DONE", "error": None})

        self.assertTrue(any("job-999" in m for m in self._messages(st, "success")))

    def test_a_failed_job_is_an_error_not_a_success(self):
        st, _, _ = self._render(
            session={"training_job_id": "job-999"},
            job_status={"state": "DONE", "error": "Column user_rating not found"})

        self.assertFalse(self._messages(st, "success"),
                         "a failed training run reported success")
        self.assertTrue(any("user_rating" in m for m in self._messages(st, "error")))

    def test_the_outcome_survives_the_next_rerun(self):
        """Streamlit reruns constantly. An outcome shown once and lost tells
        the user nothing ten seconds later."""
        _, _, state = self._render(
            session={"training_job_id": "job-999"},
            job_status={"state": "DONE", "error": "boom"})

        st, _, _ = self._render(session=state)

        self.assertTrue(any("boom" in m for m in self._messages(st, "error")))

    def test_a_remembered_outcome_is_not_re_polled(self):
        st, _, _ = self._render(session={"training_last_outcome":
                                         {"job_id": "job-999", "error": None}})

        st.status_lookup.assert_not_called()


if __name__ == "__main__":
    unittest.main()
