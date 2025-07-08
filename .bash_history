clear
ls
pwd
wget --help head -10

wget -r -np -nH --cut-dirs=0 -R "index.html*" https://cloud.tsinghua.edu.cn/d/ae84127529a24fb6b669/
wget -r -np -nH --cut-dirs=0 -R "index.html*" https://cloud.tsinghua.edu.cn/f/5989155f8bf54c6393d7/?dl=1
wget https://cloud.tsinghua.edu.cn/f/5989155f8bf54c6393d7/?dl=1
wget -0 "MAGIC" "https://cloud.tsinghua.edu.cn/d/5eb1b6b7478a479baf5az/"
wget -O "MAGIC" "https://cloud.tsinghua.edu.cn/d/5eb1b6b7478a479baf5az/"
wget -O "MAGIC.zip" "https://cloud.tsinghua.edu.cn/f/f8b1983c3c624f5da9de/?dl=1"
unzip MAGIC.zip
python3 -m zipfile -e MAGIC.zip .
wget -r -np -nH --cut-dirs=0 -R "index.html*" https://cloud.tsinghua.edu.cn/f/5989155f8bf54c6393d7/?dl=1
wget -O "MAGIC.zip" "https://cloud.tsinghua.edu.cn/f/f8b1983c3c624f5da9de/?dl=1"
wget -r -np -nH --cut-dirs=0 -R "index.html*" https://cloud.tsinghua.edu.cn/f/5989155f8bf54c6393d7/?dl=1wget -O "MAGIC.zip" "https://cloud.tsinghua.edu.cn/f/f8b1983c3c624f5da9de/?dl=1"
pip3 install unzip
apt install python3-pip
python3 -m zipfile -e MAGIC.zip .
scp "D:\papercode\MAGIC.zip" liuxy@admin:/home/liuxy/
、、、
. "/home/liuxy/.cursor-server/bin/7111807980fa9c93aedd455ffa44b682c0dc1350/out/vs/workbench/contrib/terminal/common/scripts/shellIntegration-bash.sh"
python3 -m zipfile -e MAGIC.zip .
file MAGIC.zip
hexdump -C MAGIC.zip | head -5
ls -lh MAGIC.zip
which 7z || which unzip || echo "No unzip tools found"
python3 -c "import zipfile; z = zipfile.ZipFile('MAGIC.zip'); print('Files in zip:', len(z.namelist())); print('First few files:', z.namelist()[:5] if z.namelist() else 'No files')"
python3 -c "import zipfile; z = zipfile.ZipFile('MAGIC.zip'); z.extractall('.'); print('Extraction completed')"
python3 -c "
import zipfile
import os
z = zipfile.ZipFile('MAGIC.zip')
for name in z.namelist():
    try:
        z.extract(name, '.')
        print(f'Extracted: {name}')
    except Exception as e:
        print(f'Failed to extract {name}: {e}')
        continue
print('Extraction completed')
"
find MAGIC -name "*.md" -o -name "README*" -o -name "readme*"
tree MAGIC/MAGIC -L 2
find MAGIC/MAGIC -type d | head -20
find MAGIC/MAGIC -name "*.py" | head -20
python3 -c "
import zipfile
z = zipfile.ZipFile('MAGIC.zip')
for name in z.namelist():
    if name.startswith('MAGIC/MAGIC/model/') and name.endswith('.py'):
        try:
            z.extract(name, '.')
            print(f'Extracted: {name}')
        except Exception as e:
            print(f'Failed to extract {name}: {e}')
"
python3 -c "import zipfile; z = zipfile.ZipFile('MAGIC.zip'); print('Total files:', len(z.namelist())); print('Test zip integrity...'); z.testzip()"
python3 -c "
import zipfile
import os
z = zipfile.ZipFile('MAGIC.zip')
for name in z.namelist():
    if name.startswith('MAGIC/MAGIC/model/') and name.endswith('.py'):
        try:
            # 创建目录
            os.makedirs(os.path.dirname(name), exist_ok=True)
            # 尝试读取文件内容
            content = z.read(name)
            # 写入文件
            with open(name, 'wb') as f:
                f.write(content)
            print(f'Successfully extracted: {name}')
        except Exception as e:
            print(f'Failed to extract {name}: {e}')
"
python3 -c "
import zipfile
z = zipfile.ZipFile('MAGIC.zip')
for name in z.namelist():
    if name.startswith('MAGIC/MAGIC/model/') and name.endswith('.py'):
        try:
            info = z.getinfo(name)
            print(f'{name}: size={info.file_size}, compressed={info.compress_size}, method={info.compress_type}')
        except Exception as e:
            print(f'{name}: Error getting info - {e}')
"
python3 -c "
import zipfile
z = zipfile.ZipFile('MAGIC.zip')
for name in z.namelist():
    if name.startswith('MAGIC/MAGIC/model/') and name.endswith('.py'):
        try:
            content = z.read(name)
            print(f'{name}: {len(content)} bytes')
            if len(content) < 100:
                print(f'  Content: {repr(content)}')
        except Exception as e:
            print(f'{name}: Error reading - {e}')
"
rm -rf MAGIC && python3 -m zipfile -e MAGIC.zip .
python3 -c "
import zipfile
import os
z = zipfile.ZipFile('MAGIC.zip')
success_count = 0
failed_count = 0
for name in z.namelist():
    try:
        z.extract(name, '.')
        success_count += 1
        if success_count % 100 == 0:
            print(f'Successfully extracted {success_count} files...')
    except Exception as e:
        failed_count += 1
        if failed_count <= 10:  # 只显示前10个失败的文件
            print(f'Failed to extract {name}: {e}')
        elif failed_count == 11:
            print('... (more failures)')
print(f'Extraction completed: {success_count} successful, {failed_count} failed')
"
. "/home/liuxy/.cursor-server/bin/7111807980fa9c93aedd455ffa44b682c0dc1350/out/vs/workbench/contrib/terminal/common/scripts/shellIntegration-bash.sh"
curl -L "https://cloud.tsinghua.edu.cn/d/5eb1b6b7478a479baf5a/" | head -50
curl -L "https://cloud.tsinghua.edu.cn/d/5eb1b6b7478a479baf5a/" > temp_page2.html
curl -c cookies2.txt -b cookies2.txt -L "https://cloud.tsinghua.edu.cn/d/5eb1b6b7478a479baf5a/"
curl -c cookies2.txt -b cookies2.txt -H "X-CSRFToken: ZnY6eUGCR3G5IhOtkWPajX86vp9qfpu40YXX1V3l24740laD5qXqMuQOEVQ3QLf7" -H "Accept: application/json" "https://cloud.tsinghua.edu.cn/api/v2.1/share-links/5eb1b6b7478a479baf5a/dir/?path=/"
curl -c cookies2.txt -b cookies2.txt -H "X-CSRFToken: ZnY6eUGCR3G5IhOtkWPajX86vp9qfpu40YXX1V3l24740laD5qXqMuQOEVQ3QLf7" "https://cloud.tsinghua.edu.cn/d/5eb1b6b7478a479baf5a/download/" -o MAGIC_folder.zip
file MAGIC_folder.zip
. "/home/liuxy/.cursor-server/bin/7111807980fa9c93aedd455ffa44b682c0dc1350/out/vs/workbench/contrib/terminal/common/scripts/shellIntegration-bash.sh"
wget -O "模电-综合论文.zip" "https://cloud.tsinghua.edu.cn/f/5989155f8bf54c6393d7/?dl=1"
ls -la *.zip && file "模电-综合论文.zip"
unzip "模电-综合论文.zip"
python3 -m zipfile -e "模电-综合论文.zip" .
. "/home/liuxy/.cursor-server/bin/7111807980fa9c93aedd455ffa44b682c0dc1350/out/vs/workbench/contrib/terminal/common/scripts/shellIntegration-bash.sh"
wget -r -np -nH --cut-dirs=0 -R "index.html*" https://cloud.tsinghua.edu.cn/library/961ecdb2-62a2-4167-80d7-eefe5033bd9a/MAGIC/
curl -L "https://cloud.tsinghua.edu.cn/library/961ecdb2-62a2-4167-80d7-eefe5033bd9a/MAGIC/" | grep -o 'href="[^"]*"' | grep -v "\.\./" | sed 's/href="//g' | sed 's/"//g'
wget -r -np -nH --cut-dirs=0 -R "index.html*" https://cloud.tsinghua.edu.cn/d/ae84127529a24fb6b669/
curl -L "https://cloud.tsinghua.edu.cn/d/ae84127529a24fb6b669/" | head -50
curl -L "https://cloud.tsinghua.edu.cn/d/ae84127529a24fb6b669/" > temp_page.html
curl -H "Accept: application/json" "https://cloud.tsinghua.edu.cn/api/v2.1/share-links/ae84127529a24fb6b669/dir/" | jq .
sudo apt update && sudo apt install -y jq
curl -H "Accept: application/json" "https://cloud.tsinghua.edu.cn/api/v2.1/share-links/ae84127529a24fb6b669/dir/"
curl -H "Accept: application/json" "https://cloud.tsinghua.edu.cn/api/v2.1/share-links/ae84127529a24fb6b669/dir/?path=/"
curl -H "Accept: application/json" "https://cloud.tsinghua.edu.cn/seafhttp/files/ae84127529a24fb6b669/"
curl -L "https://cloud.tsinghua.edu.cn/d/ae84127529a24fb6b669/download/" -o MAGIC.zip
ls -la MAGIC.zip && file MAGIC.zip
curl -c cookies.txt -b cookies.txt -L "https://cloud.tsinghua.edu.cn/d/ae84127529a24fb6b669/"
curl -c cookies.txt -b cookies.txt -H "X-CSRFToken: lrZQax9EkjQgUBWRAkSi6Li02KBc02kNhJGIh6ZQ1auUJXnAQiyhNWDzcx9fC7p3" -H "Accept: application/json" "https://cloud.tsinghua.edu.cn/api/v2.1/share-links/ae84127529a24fb6b669/dir/?path=/"
curl -c cookies.txt -b cookies.txt -H "X-CSRFToken: lrZQax9EkjQgUBWRAkSi6Li02KBc02kNhJGIh6ZQ1auUJXnAQiyhNWDzcx9fC7p3" "https://cloud.tsinghua.edu.cn/d/ae84127529a24fb6b669/download/" -o MAGIC_download.zip
file MAGIC_download.zip
curl -c cookies.txt -b cookies.txt -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" -H "Referer: https://cloud.tsinghua.edu.cn/d/ae84127529a24fb6b669/" "https://cloud.tsinghua.edu.cn/seafhttp/files/ae84127529a24fb6b669/"
. "/home/liuxy/.cursor-server/bin/7111807980fa9c93aedd455ffa44b682c0dc1350/out/vs/workbench/contrib/terminal/common/scripts/shellIntegration-bash.sh"
git clone https://github.com/abrahaamm/magic2.git
. "/home/liuxy/.cursor-server/bin/7111807980fa9c93aedd455ffa44b682c0dc1350/out/vs/workbench/contrib/terminal/common/scripts/shellIntegration-bash.sh"
git clone https://github.com/abrahaamm/magic2.git magic2_repo
ping -c 4 github.com
curl -I https://github.com
. "/home/liuxy/.cursor-server/bin/7111807980fa9c93aedd455ffa44b682c0dc1350/out/vs/workbench/contrib/terminal/common/scripts/shellIntegration-bash.sh"
ls -la "index.html?dl=1" && file "index.html?dl=1"
which python3 && python3 --version
echo $VIRTUAL_ENV
ls -la ~/ | grep -E "(venv|env|\.virtualenv|\.venv)"
find /home -name "*venv*" -o -name "*env*" 2>/dev/null | head -10
which conda && conda env list
pip list | grep -E "(torch|dgl|scikit-learn)"
python3 --version
sudo apt update && sudo apt install -y python3-pip
which pip3 || which pip || echo "pip not found"
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py --user
export PATH="$HOME/.local/bin:$PATH" && pip --version
export PATH="$HOME/.local/bin:$PATH" && pip install --user -r requirements.txt
export PATH="$HOME/.local/bin:$PATH" && python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
export PATH="$HOME/.local/bin:$PATH" && python3 -c "import dgl; print(f'DGL: {dgl.__version__}')"
export PATH="$HOME/.local/bin:$PATH" && python3 -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"
export PATH="$HOME/.local/bin:$PATH" && pip install --user "numpy<2.0"
export PATH="$HOME/.local/bin:$PATH" && python3 -c "import torch, dgl, sklearn; print(f'PyTorch: {torch.__version__}'); print(f'DGL: {dgl.__version__}'); print(f'Scikit-learn: {sklearn.__version__}')"
export PATH="$HOME/.local/bin:$PATH" && cd model && python3 train.py --dataset trace
export PATH="$HOME/.local/bin:$PATH" && cd .. && python3 analyze.py
. "/home/liuxy/.cursor-server/bin/7111807980fa9c93aedd455ffa44b682c0dc1350/out/vs/workbench/contrib/terminal/common/scripts/shellIntegration-bash.sh"
cd /home/liuxy/magic2 && python3 -c "
from utils.loaddata import load_metadata, load_entity_level_dataset
metadata = load_metadata('trace')
print('节点特征维度:', metadata['node_feature_dim'])
print('边特征维度:', metadata['edge_feature_dim'])
g = load_entity_level_dataset('trace', 'train', 0)
print('第一个图节点数:', g.number_of_nodes())
print('节点特征形状:', g.ndata['attr'].shape)
print('节点特征样本:', g.ndata['attr'][:3])
node_types = g.ndata['attr'].argmax(dim=1)
print('节点类型分布:', {i: (node_types == i).sum().item() for i in range(11)})
"
. "/home/liuxy/.cursor-server/bin/7111807980fa9c93aedd455ffa44b682c0dc1350/out/vs/workbench/contrib/terminal/common/scripts/shellIntegration-bash.sh"
ls -la data/trace/
python train.py --dataset trace
ls -la checkpoints/
python3 check_model_config.py
python3 eval.py --dataset trace
python3 train.py --dataset trace
python3 eval.py --dataset trace
wget https://cloud.tsinghua.edu.cn/f/108b1af790bc4e4eb8dc/?dl=1
python3 -m zipfile -e "index.html?dl=1" .
cd model/
python3 train.py --dataset trace
python3 --info
python3 --h
python3 -h
export PATH="$HOME/.local/bin:$PATH"
python3 train.py --dataset trace
cd ..
python3 train.py --dataset trace
python3 eval.py --dataset trace
python3 train.py --dataset trace
python3 eval.py --dataset trace
. "/home/liuxy/.cursor-server/bin/7111807980fa9c93aedd455ffa44b682c0dc1350/out/vs/workbench/contrib/terminal/common/scripts/shellIntegration-bash.sh"
env | grep -i proxy; git config --global --get http.proxy; git config --global --get https.proxy
git clone https://github.com/abrahaamm/magic2.git
ls -la magic2
cd ..
. "/home/liuxy/.cursor-server/bin/7111807980fa9c93aedd455ffa44b682c0dc1350/out/vs/workbench/contrib/terminal/common/scripts/shellIntegration-bash.sh"
ls -la MAGIC*.zip
cd /home/liuxy && tar -xzf MAGIC.tar.gz
