# backup.py
import os
import shutil
import sys


def save_script_backup(script_path, backup_dir):
    """
    Function to save a backup of a script to a specified backup directory.

    Args:
        script_path (str): The absolute or relative path to the script to be backed up.
        backup_dir (str): The directory where the script backup will be saved.

    Returns:
        None
    """
    script_path = os.path.abspath(script_path)
    script_rel_path = os.path.relpath(script_path, start=os.getcwd())
    backup_script_path = os.path.join(backup_dir, script_rel_path)

    os.makedirs(os.path.dirname(backup_script_path), exist_ok=True)
    shutil.copy2(script_path, backup_script_path)
    print(f"Backup of the script saved to: {backup_script_path}")


def backup_imported_modules(project_root, backup_dir="backups"):
    """
    Backup imported Python module files in a project to a specified directory.

    This function iterates through all imported modules and attempts to back up
    their source files (.py files) if they are part of a specified project. It ensures
    that only files within the specified project root directory are backed up. The
    backups are organized in a mirrored directory structure inside the backup
    directory.

    Parameters:
        project_root: str
            The absolute or relative path to the root directory of the project.
            Only modules with files inside this directory will be backed up.
        backup_dir: str
            The absolute or relative path to the directory where the backup
            files will be stored. The default value is "backups".

    Raises:
        Exception
            Any exception raised during the backup process for individual
            module files is caught and a warning is printed instead of
            raising it further.

    """
    project_root = os.path.abspath(project_root)

    for module_name, module in sys.modules.items():
        try:
            module_file = getattr(module, '__file__', None)
            if module_file and module_file.endswith('.py'):
                module_file = os.path.abspath(module_file)
                # Only backup files within the project root
                if module_file.startswith(project_root):
                    module_rel_path = os.path.relpath(module_file, start=project_root)
                    backup_module_path = os.path.join(backup_dir, module_rel_path)

                    os.makedirs(os.path.dirname(backup_module_path), exist_ok=True)
                    shutil.copy2(module_file, backup_module_path)
                    print(f"Backup of the module {module_name} saved to: {backup_module_path}")
        except Exception as e:
            print(f"Failed to backup module {module_name}: {e}")
