1. unsloth 특정 버전에 맞게 설치하기 

```angular2html
pip install torch==2.5.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
```

- `2.5 version`으로 설치 해야 정상 설치 가능 
- 설치 후 

```angular2html
python _auto_install.py
```

```angular2html
(venv) icucheol@instance-20250217-105608:~/Finance-Model/model$ python _auto_install.py 
pip install --upgrade pip && pip install "unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git"
```

하기 명령어로 본인 컴퓨터에 맞는 `unsloth` 버전을 설치하라는 명령이 떨어짐 

그에 맞게 설치해야함! 필! 

그렇지 않으면 학습이 제대로 되지 않으며 컴퓨팅 낭비가 심함 