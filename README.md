# U2NET

💻 Coding by SunWoo(tjsntjsn20@gmail.com) <br>
💻 Last-Updated on 2022-01-09 <br>
💻 This repository is made from [[2020, PR] U2Net, Going Deeper with NEsted U-structure for salient object detection](https://arxiv.org/abs/2005.09007) <br>
⚠ Still writing the code...

## ToDo-List
- [ ] model restructure
    - [ ] hooks
    - [ ] save best-model
    - [ ] loss function
    - [ ] scheduler
    - [ ] logging : eval metrics
    - [ ] build testor
    - [ ] ddp-model
    - [ ] TrainerBase : ABCMeta
- [ ] data
    - [ ] data-loader : mapping
- [ ] launch
- [ ] evaluation : add measures
- [ ] training test
- [ ] create demo script
- [x] hydra : config(overrides), logger

<br>

## 🔥 Run
- **Create your own config-file in the configs/user directory.**
- You can find a config-file example with the name configs/user/1st_training.yaml.
```{bash}
python main.py user=1st_training
```
<br>