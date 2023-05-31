# MS_Base
Hello, 
These are the pipline codes for MS classification with multimodal fusion research.

There are 2 types of models which are the baseline models(`Networks/...`) and the basic models(`Base_Model.py` and `Ori_Model`) for my future works.
They are operated by `Main.py` files which are called `Main_baseline.py` and `Main_Base.py`.

=======================================================================================
#### Info.
`Base_Model.py` 
- No additional module for Baseline
- cross-attention + fusion (m=2)

`Ori_Model.py `
- I changed tensorflow code to pytorch (reference code is written below)  
- self-attention + fusion (m=5)

### Reference
J. -U. Hwang, J. -S. Bang and S. -W. Lee, "Classification of Motion Sickness Levels using Multimodal Biosignals in Real Driving Conditions," 2022 IEEE International Conference on Systems, Man, and Cybernetics (SMC), Prague, Czech Republic, 2022, pp. 1304-1309, doi: 10.1109/SMC53654.2022.9945559.
