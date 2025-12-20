import unittest
from utils.url_generator import generate_maps_url

class TestUrlGenerator(unittest.TestCase):
    def test_generate_maps_url_basic(self):
        name = "The Great Burger"
        address = "123 High Street"
        postcode = "SW1A 1AA"
        expected = "https://www.google.com/maps/search/?api=1&query=The+Great+Burger+123+High+Street+SW1A+1AA"
        self.assertEqual(generate_maps_url(name, address, postcode), expected)

    def test_generate_maps_url_special_characters(self):
        name = "Fish & Chips @ Sea"
        address = "45 Ocean Way"
        postcode = "BN1 1EE"
        # & should be encoded to %26 or +, @ to %40 or + etc. quote_plus usually uses + for spaces.
        # urllib.parse.quote_plus("Fish & Chips @ Sea") -> 'Fish+%26+Chips+%40+Sea'
        expected = "https://www.google.com/maps/search/?api=1&query=Fish+%26+Chips+%40+Sea+45+Ocean+Way+BN1+1EE"
        self.assertEqual(generate_maps_url(name, address, postcode), expected)

    def test_generate_maps_url_missing_postcode(self):
        name = "No Postcode Cafe"
        address = "67 Hidden Lane"
        postcode = None
        expected = "https://www.google.com/maps/search/?api=1&query=No+Postcode+Cafe+67+Hidden+Lane"
        self.assertEqual(generate_maps_url(name, address, postcode), expected)

if __name__ == '__main__':
    unittest.main()
