Sayan Brahma
This Zip contains 2 Folders names phase1 and 2.
Phase 2 does not have the dataset because of the size constraints.

For Execution

In phase 1 code, the addresses of the folders containing the images has to be given inside the code.
For getting all the map ad gradient images, please change line 315 from 0 to 9. For me i have considered 10 as 2 index image. The evaluator might have to change the address of all the image folders inside the code.

In phase 2, by default the model is set for CNN,5 epochs and 5 minibatches however one may change the address in the command terminal or in the code, i would prefer changes to be made in the code.
The model for all the networks is given in the checkpoint folder.

For any kind of changes desired:
In the Train.py, select any one to train with from line 134-137.
In the Test.py, select the same from line 109-112.
And accordingly change the address of the saved model after running the train.py.
