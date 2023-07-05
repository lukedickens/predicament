# Dissesrtation
UCL dissertation

ERP_Ray.py is the main scripts for EEGNet model and main-Thin-ResNet.py is the main scripts for EEG-DL model



# By Luke

It would appear that the first bit of code to be run is the prepare_evaluation_data.py module, which creates a subfolder with train and test data that can then be used with the machine learning methods.

There are two object types which make this possible, Ray.EEG_data.EEG_data_obj (my updated name) and Ray.EEG_details.Event_time_details These almost certainly need better names.

To create Ray.EEG_data.EEG_data_obj for all participants, you need to run EEG_data.read_all_VG_files() then you need to create a set of Ray.EEG_details.Event_time_details objects so you can load in the event_details otherwise odd errors appear. See Ray.data_load_save.set_up for this process in summary.

Once you have the participants bio-sensor data and real world events loaded up, you can create a train test dataset from them with gen_EEG_train_test_to_csv. This does the splitting then saves to csv, but we really want to investigate Ray.data_load_save.gen_EEG_traintest_to_csv_mix next to work out how this data is being split.
