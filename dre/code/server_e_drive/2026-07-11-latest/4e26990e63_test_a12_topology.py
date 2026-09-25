import unittest

from a12_vcsn.topology_features import parse_channel_topology


class A12TopologyTests(unittest.TestCase):
    def test_conservative_topology_parser(self):
        self.assertEqual(parse_channel_topology("LF02REF"), ("LF", 2, True))
        self.assertEqual(parse_channel_topology("not-a-contact"), (None, None, False))

