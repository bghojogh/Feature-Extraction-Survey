1. For the given MNIST dataset, get four different CSV files - one containing training data points, second one
with training data labels, third one having all the test data points, and last one with only the test data labels.
(Make sure each file has a header line)
2. Using the MATLAB code for converting CSV files into .mat, get four different MATLAB tables one for each of the 
CSV files.
The function to convert the training/test data points CSV files to .mat tables is convert_csv_to_mat_feat(filename)
The function to convert the training/test data labels CSV files to .mat tables is convert_csv_to_mat_labels(filename)
Use the following code to get the .mat tables:
[fea_tr] = convert_csv_to_mat_feat('MNIST_train.csv');
[fea_tst] = convert_csv_to_mat_feat('MNIST_test.csv');
[gnd_tr] = convert_csv_to_mat_label('MNIST_train_labels.csv');
[gnd_tst] = convert_csv_to_mat_label('MNIST_test_labels.csv');
The above mentioned code can be run from a test program in the MATLAB file titled, "getting_mat_files_for_mRMR.m"
3. Once the .mat tables are generated, save them in the path where the current code for mRMR is present.
4. Make sure the package skfeatures is intalled 
5. Run the python code to execute mRMR method