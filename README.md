# AI

## Description  
한국어 텍스트에 대한 감성 분류 모델. 과학기술정보통신부가 주관하고 미디어젠이 수행한 '인공지능 학습용 데이터 구축사업'의 [감성 대화 말뭉치](https://aihub.or.kr/aidata/7978)를 사용하는 것을 전제로 제작되었으며 다음 60개 클래스 중 하나로 분류됨.  
|분류|E10|E20|E30|E40|E50|E60|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|0|분노|슬픔|불안|상처|당황|기쁨|  
|1|툴툴대는|실망한|두려운|질투하는|**고립된**|감사하는|
|2|좌절한|비통한|스트레스 받는|배신당한|남의 시선 의식|사랑하는|
|3|짜증나는|후회되는|취약한|**고립된**|외로운|편안한|
|4|방어적인|우울한|**혼란스러운**|충격받은|열등감|만족스러운|
|5|악의적인|마비된|당혹스러운|불우한|죄책감|흥분되는|
|6|안달하는|염세적인|회의적인|희생된|부끄러운|느긋한|
|7|구역질나는|눈물나는|걱정스러운|억울한|혐오스러운|안도하는|
|8|노여워하는|낙담한|조심스러운|괴로워하는|한심한|신이 난|
|9|성가신|환멸을 느끼는|초조한|버려진|**혼란스러운**|자신하는|


## Prerequisite
* [PyTorch](https://pytorch.org/get-started/locally/) == 1.11.0+cu102
* [Transformers](https://huggingface.co/docs/transformers/installation) == 4.19.2
* [KoBERT](https://github.com/SKTBrain/KoBERT)
* [KoELECTRA](https://github.com/monologg/KoELECTRA)
* [Pandas](https://pandas.pydata.org/getting_started.html)


## Usage
```
python train.py --batch_size 16 --epoch 10 --freeze_layer 9 --model mymodel.pt
```
Relative Import 에러의 경우 다음과 같이 실행  
```
python -m AI.kobert.train --batch_size 16 --epoch 10 --freeze_layer 9 --model mymodel.pt
```
