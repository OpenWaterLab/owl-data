import owl.data
from owl.data.dataset import Dataset
from owl.data.sensor_dataset import SensorDataset
from owl.data.lab_dataset import LabDataset


def test_package_import():
	assert owl.data is not None
	


def test_class_imports():
	assert Dataset is not None
    assert SensorDataset is not None
    assert LabDataset is not None