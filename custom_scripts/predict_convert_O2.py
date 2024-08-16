import tifffile
import h5py
import os



def convert_ome_tiff_to_h5(ome_tiff_path, h5_directory, h5_name, channels=None):
    """
    Convert an OME-TIFF file to an HDF5 file and save only specified channels.
    
    Parameters:
    ome_tiff_path (str): Path to the input OME-TIFF file.
    h5_directory (str): Directory to save the output HDF5 file.
    h5_name (str): Name of the output HDF5 file.
    channels (list, optional): List of channel indices to save. If None, all channels are saved.
    """
    # Ensure the output directory exists
    os.makedirs(h5_directory, exist_ok=True)

    # Read the OME-TIFF file
    ome_tiff_data = tifffile.imread(ome_tiff_path)

    # Define the output HDF5 file path
    h5_file_path = os.path.join(h5_directory, h5_name + '.h5')

    # If channels is None, save all channels
    if channels is None:
        selected_data = ome_tiff_data
    else:
        # Validate the channels
        if ome_tiff_data.ndim == 3:
            max_channels = ome_tiff_data.shape[0]
            for channel in channels:
                if channel < 0 or channel >= max_channels:
                    raise ValueError(f"Invalid channel index {channel}. It should be between 0 and {max_channels - 1}.")
            selected_data = ome_tiff_data[channels, :, :]
        elif ome_tiff_data.ndim == 2:
            if len(channels) > 1 or channels[0] != 0:
                raise ValueError("Invalid channel selection for a single channel image.")
            selected_data = ome_tiff_data
        else:
            raise ValueError(f"Unexpected data shape: {ome_tiff_data.shape}")

    # Write the selected data to an HDF5 file
    with h5py.File(h5_file_path, 'w') as h5_file:
        h5_file.create_dataset('raw', data=selected_data)

    print(f"Selected channels successfully saved to {h5_file_path}")


# running this on O2
ome_tiff_path = '/n/scratch/users/a/ajn16/cspot_new/testing/registration/image.tif'
h5_directory = '/n/scratch/users/a/ajn16/cspot_new/testing'
h5_name = 'predict'
channels = None

convert_ome_tiff_to_h5(ome_tiff_path=ome_tiff_path,h5_directory=h5_directory,h5_name=h5_name,channels=channels)
