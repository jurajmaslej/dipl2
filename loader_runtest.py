import unittest

from loader_refact import Loader_test


class MainTest(unittest.TestCase):

    def test_atomics(self):
		l = Loader_test('bright2/list_9to8.txt', 'bright2', 'pano_cropped_hsv2', new_load = False, new_save=False, w=224, h=224)
		l.create_dict_photo_synop()
		#l.dict_to_dframe()
		self.assertEqual(list(l.dict_to_dframe().columns), [u'date', u'name', u'synop'])
		self.assertEqual(l.dict_to_dframe().shape, (2714,3))

if __name__ == "__main__":
    unittest.main()