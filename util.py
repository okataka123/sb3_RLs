import os
import shutil

class Util:
    def check_and_clean_directory(path):
        '''
        指定されたディレクトリが存在するか確認し、
        - 存在する場合: 中のファイルをすべて削除
        - 存在しない場合: ディレクトリを作成

        Args:
            path (str): ディレクトリのパス
        '''
        if os.path.exists(path):
            # ディレクトリが存在する場合、中のファイルを削除
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # ファイルやシンボリックリンクを削除
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # サブディレクトリを削除
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        else:
            # ディレクトリが存在しない場合、作成
            os.makedirs(path)