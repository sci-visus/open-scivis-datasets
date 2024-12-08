# The links are manually entered instead of getting them from the datasets.json file to avoid
#   renaming/deleting a dataset (and updating datasets.json) without noticing it broke a link.
#   Raw links are the only form of links we commit to maintain indefinitely.

import hashlib
import io
import json
import numpy as np
import OpenVisus as ov
import requests
import unittest


def get_raw_links1(url):
    return [
        f"{url}/3d_neurons_15_sept_2016/3d_neurons_15_sept_2016_2048x2048x1718_uint16.raw",
        f"{url}/aneurism/aneurism_256x256x256_uint8.raw",
        f"{url}/backpack/backpack_512x512x373_uint16.raw",
        f"{url}/beechnut/beechnut_1024x1024x1546_uint16.raw",
        f"{url}/blunt_fin/blunt_fin_256x128x64_uint8.raw",
        f"{url}/bonsai/bonsai_256x256x256_uint8.raw",
        f"{url}/boston_teapot/boston_teapot_256x256x178_uint8.raw",
        f"{url}/bunny/bunny_512x512x361_uint16.raw",
        f"{url}/carp/carp_256x256x512_uint16.raw",
        f"{url}/chameleon/chameleon_1024x1024x1080_uint16.raw",
        f"{url}/christmas_tree/christmas_tree_512x499x512_uint16.raw",
        f"{url}/csafe_heptane/csafe_heptane_302x302x302_uint8.raw",
        f"{url}/dns/dns_10240x7680x1536_float64.raw",
        f"{url}/duct/duct_193x194x1000_float32.raw",
        f"{url}/engine/engine_256x256x128_uint8.raw",
        f"{url}/foot/foot_256x256x256_uint8.raw",
        f"{url}/frog/frog_256x256x44_uint8.raw",
        f"{url}/fuel/fuel_64x64x64_uint8.raw",
        f"{url}/hcci_oh/hcci_oh_560x560x560_float32.raw",
        f"{url}/hydrogen_atom/hydrogen_atom_128x128x128_uint8.raw",
        f"{url}/isotropic_pressure/isotropic_pressure_4096x4096x4096_float32.raw",
        f"{url}/jicf_q/jicf_q_1408x1080x1100_float32.raw",
        f"{url}/kingsnake/kingsnake_1024x1024x795_uint8.raw",
        f"{url}/lobster/lobster_301x324x56_uint8.raw",
        f"{url}/magnetic_reconnection/magnetic_reconnection_512x512x512_float32.raw",
        f"{url}/marmoset_neurons/marmoset_neurons_1024x1024x314_uint8.raw",
        f"{url}/marschner_lobb/marschner_lobb_41x41x41_uint8.raw",
        f"{url}/miranda/miranda_1024x1024x1024_float32.raw",
        f"{url}/mri_ventricles/mri_ventricles_256x256x124_uint8.raw",
        f"{url}/mri_woman/mri_woman_256x256x109_uint16.raw",
        f"{url}/mrt_angio/mrt_angio_416x512x112_uint16.raw",
        f"{url}/neghip/neghip_64x64x64_uint8.raw",
        f"{url}/neocortical_layer_1_axons/neocortical_layer_1_axons_1464x1033x76_uint8.raw",
        f"{url}/nucleon/nucleon_41x41x41_uint8.raw",
        f"{url}/pancreas/pancreas_240x512x512_int16.raw",
        f"{url}/pawpawsaurus/pawpawsaurus_958x646x1088_uint16.raw",
        f"{url}/pig_heart/pig_heart_2048x2048x2612_int16.raw",
        f"{url}/present/present_492x492x442_uint16.raw",
        f"{url}/prone/prone_512x512x463_uint16.raw",
        f"{url}/richtmyer_meshkov/richtmyer_meshkov_2048x2048x1920_uint8.raw",
        f"{url}/rotstrat_temperature/rotstrat_temperature_4096x4096x4096_float32.raw",
        f"{url}/shockwave/shockwave_64x64x512_uint8.raw",
        f"{url}/silicium/silicium_98x34x34_uint8.raw",
        f"{url}/skull/skull_256x256x256_uint8.raw",
        f"{url}/spathorhynchus/spathorhynchus_1024x1024x750_uint16.raw",
        f"{url}/stag_beetle/stag_beetle_832x832x494_uint16.raw",
        f"{url}/statue_leg/statue_leg_341x341x93_uint8.raw",
        f"{url}/stent/stent_512x512x174_uint16.raw",
        f"{url}/synthetic_truss_with_five_defects/synthetic_truss_with_five_defects_1200x1200x1200_float32.raw",
        f"{url}/tacc_turbulence/tacc_turbulence_256x256x256_float32.raw",
        f"{url}/tooth/tooth_103x94x161_uint8.raw",
        f"{url}/vertebra/vertebra_512x512x512_uint16.raw",
        f"{url}/vis_male/vis_male_128x256x256_uint8.raw",
        f"{url}/woodbranch/woodbranch_2048x2048x2048_uint16.raw",
        f"{url}/zeiss/zeiss_680x680x680_uint8.raw",
    ]


class TestRawLinks1(unittest.TestCase):
    def test_links(self):
        with open("config.json") as f:
            config = json.load(f)
        for link in get_raw_links1(config["url"]):
            print(link)
            self.assertEqual(requests.head(link, allow_redirects=True).status_code, 200)

    def test_backup_links(self):
        url = "http://open-scivis-datasets.sci.utah.edu/open-scivis-datasets"

        for link in get_raw_links1(url):
            print(link)
            self.assertEqual(requests.head(link, allow_redirects=True).status_code, 200)


class TestOpenVisus1(unittest.TestCase):
    def test_checksum(self):
        with open("config.json") as f:
            config = json.load(f)
        datasets = json.loads(requests.get(f'{config["url"]}/datasets.json').text)

        for name, dataset in datasets.items():
            # NOTE(3/29/2023): skip large datasets
            if dataset["size"][0] * dataset["size"][1] * dataset["size"][2] > 128**3:
                continue

            print(name, flush=True)
            bytes = io.BytesIO(requests.get(dataset["url"]).content)
            self.assertEqual(
                hashlib.sha512(bytes.getbuffer()).hexdigest(), dataset["sha512sum"]
            )

            data = np.frombuffer(
                bytes.getbuffer(), dtype=np.dtype(dataset["type"])
            ).reshape(dataset["size"][2], dataset["size"][1], dataset["size"][0])

            d = ov.load_dataset(f'{config["url"]}/{name}/{name}.idx')
            self.assertTrue((d.read() == data).all())


if __name__ == "__main__":
    unittest.main()
