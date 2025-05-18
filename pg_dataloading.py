import os

def find_edf_files(root_dir):
    edf_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith('.edf'):
                edf_files.append(os.path.join(dirpath, file))
                print(f"Found EDF file: {os.path.join(dirpath, file)}")
    return edf_files

def file():
    root_dir = '~/Documents/TUH_EEG_Corpus_v2.0.1/00_epilepsy/aaaaaanr/'
 # Change this to your target directory
    edf_files = find_edf_files(root_dir)
    
    print(f"Found {len(edf_files)} EDF files:")
    for file in edf_files:
        print(file)
    # Optionally, you can save the list to a file
    with open('edf_files_list.txt', 'w') as f:
        for file in edf_files:
            f.write(f"{file}\n")
    # Optionally, you can return the list of files
    return edf_files


def main():
    root_dir = '~/Documents/TUH_EEG_Corpus_v2.0.1/00_epilepsy/aaaaaanr/s006_2013/01_tcp_ar/aaaaaanr_s006_t001.edf'
    import mne
    raw = mne.io.read_raw_edf(root_dir, preload=True)

if __name__ == "__main__":
    main()