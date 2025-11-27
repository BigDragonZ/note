 # Windows/macOS/Linux 通用
python -m pip install --upgrade pip

# 格式：pip install 包名 -i 阿里云镜像地址
pip install requests -i https://mirrors.aliyun.com/pypi/simple/

pip install jupyterlab -i https://mirrors.aliyun.com/pypi/simple/


# 代码格式化（支持 Python/Markdown）
pip install jupyterlab-code-formatter black -i https://mirrors.aliyun.com/pypi/simple/
# Git 集成（版本控制笔记）
pip install jupyterlab-git -i https://mirrors.aliyun.com/pypi/simple/
# 语法补全与语法检查
pip install jupyterlab-lsp python-lsp-server -i https://mirrors.aliyun.com/pypi/simple/


pip install pipreqs
# 导出所有依赖到 requirements.txt
pip freeze > requirements.txt


pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple