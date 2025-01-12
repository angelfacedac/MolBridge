import os
import shutil
import zipfile

from src.utils.manager import DATA_SOURCE, MODEL_NAME, EXPERIMENT_NAME


def zip_project_files(output_filename, exclude_dirs=('data', 'tmp', 'logs', '.git', '.idea')):
    """
    Zip the current project directory.
    :param output_filename: The name of the zip file to create.
    :param exclude_dirs: A list of directories to exclude from the zip file.
    """
    print("Zipping project files...")

    project_root = os.getcwd()
    print('zip工作目录：', project_root)

    temp_dir = os.path.join(project_root, "temp_to_zip")
    os.makedirs(temp_dir, exist_ok=True)
    exclude_dirs = list(exclude_dirs)
    exclude_dirs.append("temp_to_zip")

    for root, dirs, files in os.walk(project_root):

        for exclude_dir in exclude_dirs:
            if exclude_dir in dirs:
                dirs.remove(exclude_dir)

        for file in files:
            src_path = os.path.join(root, file)
            rel_path = os.path.relpath(src_path, project_root)
            dest_path = os.path.join(temp_dir, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(src_path, dest_path)

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, temp_dir))

    shutil.rmtree(temp_dir)

    print("Zip file created successfully.")


if __name__ == '__main__':
    zip_project_files(
        os.path.join('logs', DATA_SOURCE, MODEL_NAME, EXPERIMENT_NAME, 'project_backup.zip'),
        exclude_dirs=['logs', '.git', '.idea']
    )
