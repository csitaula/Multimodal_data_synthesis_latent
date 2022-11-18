# running codes
from google.colab import drive
drive.mount('/content/drive/',force_remount=True)
root_dir = '/content/drive/MyDrive/2022 Chiranjibi (RA) - Multimodal Imaging/Datasets/IXI/'
data = Data(data_folder=root_dir, dataset='IXI', trim_and_downsample=False)
data.load()

input_modalities = ['T1', 'T2', 'DWI']
output_weights = {'VFlair': 1.0, 'concat': 1.0}
exp = Experiment(input_modalities, output_weights, root_dir, data, latent_dim=16, spatial_transformer=True)
exp.run(data)