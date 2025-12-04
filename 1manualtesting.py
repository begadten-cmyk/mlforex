import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
import datetime

############################################################################
# 1) FULL YEAR 2024 DATA (Jan 1 -> Dec 26)
#    EXACTLY as you provided, stored in 'train_data'.
#    Each row is { 'Date','Open','Close','High','Low','Mid' }.
#    We only use O,H,L,C in the code; 'Mid' is dropped.
############################################################################

train_data = [

    # ===================== JAN 2024 =====================
    {'Date':'2024-01-01','Open':0.8669,'Close':0.8669,'High':0.8669,'Low':0.8669,'Mid':0.8669},
    {'Date':'2024-01-02','Open':0.8670,'Close':0.8670,'High':0.8683,'Low':0.8646,'Mid':0.8664},
    {'Date':'2024-01-03','Open':0.8671,'Close':0.8623,'High':0.8676,'Low':0.8617,'Mid':0.8647},
    {'Date':'2024-01-04','Open':0.8624,'Close':0.8632,'High':0.8640,'Low':0.8610,'Mid':0.8625},
    {'Date':'2024-01-05','Open':0.8631,'Close':0.8604,'High':0.8635,'Low':0.8600,'Mid':0.8617},
    {'Date':'2024-01-08','Open':0.8605,'Close':0.8589,'High':0.8621,'Low':0.8589,'Mid':0.8605},
    {'Date':'2024-01-09','Open':0.8590,'Close':0.8600,'High':0.8608,'Low':0.8587,'Mid':0.8597},
    {'Date':'2024-01-10','Open':0.8601,'Close':0.8611,'High':0.8621,'Low':0.8596,'Mid':0.8608},
    {'Date':'2024-01-11','Open':0.8611,'Close':0.8597,'High':0.8618,'Low':0.8594,'Mid':0.8606},
    {'Date':'2024-01-12','Open':0.8597,'Close':0.8588,'High':0.8607,'Low':0.8586,'Mid':0.8596},
    {'Date':'2024-01-15','Open':0.8596,'Close':0.8603,'High':0.8612,'Low':0.8587,'Mid':0.8599},
    {'Date':'2024-01-16','Open':0.8604,'Close':0.8607,'High':0.8620,'Low':0.8591,'Mid':0.8606},
    {'Date':'2024-01-17','Open':0.8606,'Close':0.8583,'High':0.8619,'Low':0.8568,'Mid':0.8593},
    {'Date':'2024-01-18','Open':0.8585,'Close':0.8559,'High':0.8592,'Low':0.8555,'Mid':0.8573},
    {'Date':'2024-01-19','Open':0.8560,'Close':0.8579,'High':0.8593,'Low':0.8557,'Mid':0.8575},
    {'Date':'2024-01-22','Open':0.8582,'Close':0.8563,'High':0.8585,'Low':0.8555,'Mid':0.8570},
    {'Date':'2024-01-23','Open':0.8563,'Close':0.8554,'High':0.8565,'Low':0.8547,'Mid':0.8556},
    {'Date':'2024-01-24','Open':0.8555,'Close':0.8554,'High':0.8561,'Low':0.8536,'Mid':0.8549},
    {'Date':'2024-01-25','Open':0.8554,'Close':0.8534,'High':0.8565,'Low':0.8521,'Mid':0.8543},
    {'Date':'2024-01-26','Open':0.8535,'Close':0.8543,'High':0.8548,'Low':0.8525,'Mid':0.8537},
    {'Date':'2024-01-29','Open':0.8542,'Close':0.8524,'High':0.8545,'Low':0.8513,'Mid':0.8529},
    {'Date':'2024-01-30','Open':0.8524,'Close':0.8540,'High':0.8568,'Low':0.8517,'Mid':0.8542},
    {'Date':'2024-01-31','Open':0.8540,'Close':0.8527,'High':0.8548,'Low':0.8523,'Mid':0.8536},

    # ===================== FEB 2024 =====================
    {'Date':'2024-02-01','Open':0.8526,'Close':0.8531,'High':0.8559,'Low':0.8521,'Mid':0.8540},
    {'Date':'2024-02-02','Open':0.8531,'Close':0.8541,'High':0.8548,'Low':0.8521,'Mid':0.8534},
    {'Date':'2024-02-05','Open':0.8535,'Close':0.8568,'High':0.8572,'Low':0.8529,'Mid':0.8551},
    {'Date':'2024-02-06','Open':0.8569,'Close':0.8537,'High':0.8572,'Low':0.8533,'Mid':0.8553},
    {'Date':'2024-02-07','Open':0.8537,'Close':0.8531,'High':0.8541,'Low':0.8516,'Mid':0.8529},
    {'Date':'2024-02-08','Open':0.8531,'Close':0.8541,'High':0.8545,'Low':0.8530,'Mid':0.8537},
    {'Date':'2024-02-09','Open':0.8542,'Close':0.8540,'High':0.8547,'Low':0.8532,'Mid':0.8540},
    {'Date':'2024-02-12','Open':0.8537,'Close':0.8531,'High':0.8547,'Low':0.8528,'Mid':0.8538},
    {'Date':'2024-02-13','Open':0.8530,'Close':0.8505,'High':0.8536,'Low':0.8500,'Mid':0.8518},
    {'Date':'2024-02-14','Open':0.8504,'Close':0.8538,'High':0.8549,'Low':0.8498,'Mid':0.8524},
    {'Date':'2024-02-15','Open':0.8538,'Close':0.8550,'High':0.8570,'Low':0.8536,'Mid':0.8553},
    {'Date':'2024-02-16','Open':0.8550,'Close':0.8552,'High':0.8565,'Low':0.8539,'Mid':0.8552},
    {'Date':'2024-02-19','Open':0.8552,'Close':0.8558,'High':0.8560,'Low':0.8538,'Mid':0.8549},
    {'Date':'2024-02-20','Open':0.8558,'Close':0.8563,'High':0.8578,'Low':0.8546,'Mid':0.8562},
    {'Date':'2024-02-21','Open':0.8562,'Close':0.8561,'High':0.8572,'Low':0.8557,'Mid':0.8565},
    {'Date':'2024-02-22','Open':0.8563,'Close':0.8549,'High':0.8576,'Low':0.8546,'Mid':0.8561},
    {'Date':'2024-02-23','Open':0.8548,'Close':0.8538,'High':0.8552,'Low':0.8528,'Mid':0.8540},
    {'Date':'2024-02-26','Open':0.8536,'Close':0.8554,'High':0.8561,'Low':0.8533,'Mid':0.8547},
    {'Date':'2024-02-27','Open':0.8555,'Close':0.8549,'High':0.8567,'Low':0.8548,'Mid':0.8558},
    {'Date':'2024-02-28','Open':0.8549,'Close':0.8560,'High':0.8566,'Low':0.8546,'Mid':0.8556},
    {'Date':'2024-02-29','Open':0.8560,'Close':0.8559,'High':0.8570,'Low':0.8551,'Mid':0.8561},

    # ===================== MAR 2024 =====================
    {'Date':'2024-03-01','Open':0.8559,'Close':0.8565,'High':0.8576,'Low':0.8555,'Mid':0.8565},
    {'Date':'2024-03-04','Open':0.8565,'Close':0.8553,'High':0.8568,'Low':0.8550,'Mid':0.8559},
    {'Date':'2024-03-05','Open':0.8554,'Close':0.8545,'High':0.8561,'Low':0.8535,'Mid':0.8548},
    {'Date':'2024-03-06','Open':0.8545,'Close':0.8561,'High':0.8563,'Low':0.8541,'Mid':0.8552},
    {'Date':'2024-03-07','Open':0.8561,'Close':0.8548,'High':0.8563,'Low':0.8524,'Mid':0.8544},
    {'Date':'2024-03-08','Open':0.8548,'Close':0.8511,'High':0.8549,'Low':0.8504,'Mid':0.8526},
    {'Date':'2024-03-11','Open':0.8510,'Close':0.8526,'High':0.8535,'Low':0.8509,'Mid':0.8522},
    {'Date':'2024-03-12','Open':0.8526,'Close':0.8542,'High':0.8555,'Low':0.8523,'Mid':0.8539},
    {'Date':'2024-03-13','Open':0.8542,'Close':0.8555,'High':0.8560,'Low':0.8539,'Mid':0.8549},
    {'Date':'2024-03-14','Open':0.8555,'Close':0.8534,'High':0.8559,'Low':0.8534,'Mid':0.8546},
    {'Date':'2024-03-15','Open':0.8534,'Close':0.8550,'High':0.8554,'Low':0.8533,'Mid':0.8543},
    {'Date':'2024-03-18','Open':0.8545,'Close':0.8541,'High':0.8563,'Low':0.8540,'Mid':0.8551},
    {'Date':'2024-03-19','Open':0.8541,'Close':0.8541,'High':0.8556,'Low':0.8532,'Mid':0.8544},
    {'Date':'2024-03-20','Open':0.8541,'Close':0.8543,'High':0.8556,'Low':0.8536,'Mid':0.8546},
    {'Date':'2024-03-21','Open':0.8543,'Close':0.8579,'High':0.8583,'Low':0.8530,'Mid':0.8557},
    {'Date':'2024-03-22','Open':0.8579,'Close':0.8578,'High':0.8602,'Low':0.8568,'Mid':0.8585},
    {'Date':'2024-03-25','Open':0.8582,'Close':0.8576,'High':0.8585,'Low':0.8565,'Mid':0.8575},
    {'Date':'2024-03-26','Open':0.8577,'Close':0.8578,'High':0.8589,'Low':0.8569,'Mid':0.8579},
    {'Date':'2024-03-27','Open':0.8578,'Close':0.8566,'High':0.8586,'Low':0.8565,'Mid':0.8575},
    {'Date':'2024-03-28','Open':0.8566,'Close':0.8546,'High':0.8571,'Low':0.8544,'Mid':0.8558},
    {'Date':'2024-03-29','Open':0.8546,'Close':0.8553,'High':0.8555,'Low':0.8530,'Mid':0.8543},

    # ===================== APR 2024 =====================
    {'Date':'2024-04-01','Open':0.8544,'Close':0.8560,'High':0.8564,'Low':0.8539,'Mid':0.8551},
    {'Date':'2024-04-02','Open':0.8560,'Close':0.8562,'High':0.8575,'Low':0.8541,'Mid':0.8558},
    {'Date':'2024-04-03','Open':0.8563,'Close':0.8564,'High':0.8582,'Low':0.8557,'Mid':0.8570},
    {'Date':'2024-04-04','Open':0.8564,'Close':0.8572,'High':0.8581,'Low':0.8562,'Mid':0.8572},
    {'Date':'2024-04-05','Open':0.8572,'Close':0.8575,'High':0.8586,'Low':0.8572,'Mid':0.8579},
    {'Date':'2024-04-08','Open':0.8583,'Close':0.8581,'High':0.8585,'Low':0.8574,'Mid':0.8579},
    {'Date':'2024-04-09','Open':0.8581,'Close':0.8564,'High':0.8583,'Low':0.8561,'Mid':0.8572},
    {'Date':'2024-04-10','Open':0.8564,'Close':0.8567,'High':0.8571,'Low':0.8548,'Mid':0.8559},
    {'Date':'2024-04-11','Open':0.8567,'Close':0.8545,'High':0.8571,'Low':0.8542,'Mid':0.8556},
    {'Date':'2024-04-12','Open':0.8545,'Close':0.8547,'High':0.8552,'Low':0.8528,'Mid':0.8540},
    {'Date':'2024-04-15','Open':0.8545,'Close':0.8537,'High':0.8553,'Low':0.8528,'Mid':0.8540},
    {'Date':'2024-04-16','Open':0.8537,'Close':0.8545,'High':0.8550,'Low':0.8528,'Mid':0.8539},
    {'Date':'2024-04-17','Open':0.8545,'Close':0.8570,'High':0.8572,'Low':0.8521,'Mid':0.8547},
    {'Date':'2024-04-18','Open':0.8570,'Close':0.8558,'High':0.8571,'Low':0.8551,'Mid':0.8561},
    {'Date':'2024-04-19','Open':0.8558,'Close':0.8615,'High':0.8616,'Low':0.8555,'Mid':0.8585},
    {'Date':'2024-04-22','Open':0.8609,'Close':0.8627,'High':0.8644,'Low':0.8608,'Mid':0.8626},
    {'Date':'2024-04-23','Open':0.8627,'Close':0.8596,'High':0.8645,'Low':0.8593,'Mid':0.8619},
    {'Date':'2024-04-24','Open':0.8596,'Close':0.8584,'High':0.8600,'Low':0.8584,'Mid':0.8592},
    {'Date':'2024-04-25','Open':0.8584,'Close':0.8575,'High':0.8592,'Low':0.8565,'Mid':0.8578},
    {'Date':'2024-04-26','Open':0.8575,'Close':0.8560,'High':0.8584,'Low':0.8558,'Mid':0.8571},
    {'Date':'2024-04-29','Open':0.8565,'Close':0.8534,'High':0.8570,'Low':0.8533,'Mid':0.8551},
    {'Date':'2024-04-30','Open':0.8534,'Close':0.8538,'High':0.8555,'Low':0.8531,'Mid':0.8543},

    # ===================== MAY 2024 =====================
    {'Date':'2024-05-01','Open':0.8538,'Close':0.8552,'High':0.8558,'Low':0.8537,'Mid':0.8547},
    {'Date':'2024-05-02','Open':0.8552,'Close':0.8557,'High':0.8565,'Low':0.8547,'Mid':0.8556},
    {'Date':'2024-05-03','Open':0.8557,'Close':0.8578,'High':0.8586,'Low':0.8551,'Mid':0.8569},
    {'Date':'2024-05-06','Open':0.8576,'Close':0.8573,'High':0.8582,'Low':0.8557,'Mid':0.8569},
    {'Date':'2024-05-07','Open':0.8573,'Close':0.8597,'High':0.8599,'Low':0.8569,'Mid':0.8584},
    {'Date':'2024-05-08','Open':0.8597,'Close':0.8600,'High':0.8618,'Low':0.8593,'Mid':0.8605},
    {'Date':'2024-05-09','Open':0.8600,'Close':0.8609,'High':0.8620,'Low':0.8591,'Mid':0.8606},
    {'Date':'2024-05-10','Open':0.8609,'Close':0.8600,'High':0.8612,'Low':0.8595,'Mid':0.8604},
    {'Date':'2024-05-13','Open':0.8602,'Close':0.8591,'High':0.8609,'Low':0.8590,'Mid':0.8600},
    {'Date':'2024-05-14','Open':0.8593,'Close':0.8592,'High':0.8614,'Low':0.8587,'Mid':0.8601},
    {'Date':'2024-05-15','Open':0.8592,'Close':0.8580,'High':0.8601,'Low':0.8576,'Mid':0.8588},
    {'Date':'2024-05-16','Open':0.8580,'Close':0.8577,'High':0.8588,'Low':0.8572,'Mid':0.8580},
    {'Date':'2024-05-17','Open':0.8577,'Close':0.8557,'High':0.8581,'Low':0.8555,'Mid':0.8568},
    {'Date':'2024-05-20','Open':0.8561,'Close':0.8544,'High':0.8568,'Low':0.8542,'Mid':0.8555},
    {'Date':'2024-05-21','Open':0.8545,'Close':0.8540,'High':0.8551,'Low':0.8534,'Mid':0.8542},
    {'Date':'2024-05-22','Open':0.8540,'Close':0.8511,'High':0.8543,'Low':0.8504,'Mid':0.8524},
    {'Date':'2024-05-23','Open':0.8511,'Close':0.8517,'High':0.8528,'Low':0.8502,'Mid':0.8515},
    {'Date':'2024-05-24','Open':0.8517,'Close':0.8516,'High':0.8532,'Low':0.8512,'Mid':0.8522},
    {'Date':'2024-05-27','Open':0.8518,'Close':0.8504,'High':0.8520,'Low':0.8497,'Mid':0.8509},
    {'Date':'2024-05-28','Open':0.8504,'Close':0.8508,'High':0.8518,'Low':0.8496,'Mid':0.8507},
    {'Date':'2024-05-29','Open':0.8508,'Close':0.8505,'High':0.8517,'Low':0.8484,'Mid':0.8500},
    {'Date':'2024-05-30','Open':0.8505,'Close':0.8508,'High':0.8517,'Low':0.8502,'Mid':0.8510},
    {'Date':'2024-05-31','Open':0.8508,'Close':0.8515,'High':0.8541,'Low':0.8501,'Mid':0.8521},

    # ===================== JUNE 2024 =====================
    {'Date':'2024-06-03','Open':0.8518,'Close':0.8514,'High':0.8536,'Low':0.8510,'Mid':0.8523},
    {'Date':'2024-06-04','Open':0.8514,'Close':0.8519,'High':0.8525,'Low':0.8506,'Mid':0.8516},
    {'Date':'2024-06-05','Open':0.8519,'Close':0.8500,'High':0.8521,'Low':0.8498,'Mid':0.8510},
    {'Date':'2024-06-06','Open':0.8500,'Close':0.8513,'High':0.8525,'Low':0.8499,'Mid':0.8512},
    {'Date':'2024-06-07','Open':0.8513,'Close':0.8493,'High':0.8521,'Low':0.8489,'Mid':0.8505},
    {'Date':'2024-06-10','Open':0.8485,'Close':0.8456,'High':0.8488,'Low':0.8440,'Mid':0.8464},
    {'Date':'2024-06-11','Open':0.8456,'Close':0.8430,'High':0.8468,'Low':0.8418,'Mid':0.8443},
    {'Date':'2024-06-12','Open':0.8430,'Close':0.8446,'High':0.8452,'Low':0.8418,'Mid':0.8435},
    {'Date':'2024-06-13','Open':0.8446,'Close':0.8413,'High':0.8458,'Low':0.8413,'Mid':0.8435},
    {'Date':'2024-06-14','Open':0.8413,'Close':0.8436,'High':0.8441,'Low':0.8397,'Mid':0.8419},
    {'Date':'2024-06-17','Open':0.8438,'Close':0.8449,'High':0.8462,'Low':0.8435,'Mid':0.8448},
    {'Date':'2024-06-18','Open':0.8449,'Close':0.8451,'High':0.8465,'Low':0.8444,'Mid':0.8455},
    {'Date':'2024-06-19','Open':0.8452,'Close':0.8446,'High':0.8456,'Low':0.8430,'Mid':0.8443},
    {'Date':'2024-06-20','Open':0.8446,'Close':0.8455,'High':0.8461,'Low':0.8434,'Mid':0.8448},
    {'Date':'2024-06-21','Open':0.8455,'Close':0.8457,'High':0.8466,'Low':0.8443,'Mid':0.8455},
    {'Date':'2024-06-24','Open':0.8458,'Close':0.8461,'High':0.8478,'Low':0.8452,'Mid':0.8465},
    {'Date':'2024-06-25','Open':0.8461,'Close':0.8445,'High':0.8465,'Low':0.8431,'Mid':0.8448},
    {'Date':'2024-06-26','Open':0.8445,'Close':0.8462,'High':0.8464,'Low':0.8433,'Mid':0.8448},
    {'Date':'2024-06-27','Open':0.8462,'Close':0.8467,'High':0.8476,'Low':0.8454,'Mid':0.8465},
    {'Date':'2024-06-28','Open':0.8467,'Close':0.8475,'High':0.8482,'Low':0.8458,'Mid':0.8470},

    # ===================== JULY 2024 =====================
    {'Date':'2024-07-01','Open':0.8493,'Close':0.8491,'High':0.8499,'Low':0.8473,'Mid':0.8486},
    {'Date':'2024-07-02','Open':0.8491,'Close':0.8471,'High':0.8496,'Low':0.8466,'Mid':0.8481},
    {'Date':'2024-07-03','Open':0.8471,'Close':0.8464,'High':0.8478,'Low':0.8459,'Mid':0.8469},
    {'Date':'2024-07-04','Open':0.8464,'Close':0.8474,'High':0.8476,'Low':0.8460,'Mid':0.8468},
    {'Date':'2024-07-05','Open':0.8474,'Close':0.8462,'High':0.8478,'Low':0.8452,'Mid':0.8465},
    {'Date':'2024-07-08','Open':0.8448,'Close':0.8452,'High':0.8460,'Low':0.8439,'Mid':0.8450},
    {'Date':'2024-07-09','Open':0.8453,'Close':0.8457,'High':0.8460,'Low':0.8442,'Mid':0.8451},
    {'Date':'2024-07-10','Open':0.8457,'Close':0.8429,'High':0.8460,'Low':0.8426,'Mid':0.8443},
    {'Date':'2024-07-11','Open':0.8428,'Close':0.8416,'High':0.8432,'Low':0.8413,'Mid':0.8423},
    {'Date':'2024-07-12','Open':0.8416,'Close':0.8393,'High':0.8422,'Low':0.8392,'Mid':0.8407},
    {'Date':'2024-07-15','Open':0.8395,'Close':0.8402,'High':0.8412,'Low':0.8389,'Mid':0.8401},
    {'Date':'2024-07-16','Open':0.8402,'Close':0.8401,'High':0.8411,'Low':0.8398,'Mid':0.8404},
    {'Date':'2024-07-17','Open':0.8401,'Close':0.8409,'High':0.8410,'Low':0.8383,'Mid':0.8397},
    {'Date':'2024-07-18','Open':0.8409,'Close':0.8418,'High':0.8423,'Low':0.8404,'Mid':0.8413},
    {'Date':'2024-07-19','Open':0.8418,'Close':0.8428,'High':0.8433,'Low':0.8413,'Mid':0.8423},
    {'Date':'2024-07-22','Open':0.8428,'Close':0.8421,'High':0.8431,'Low':0.8414,'Mid':0.8423},
    {'Date':'2024-07-23','Open':0.8421,'Close':0.8409,'High':0.8427,'Low':0.8398,'Mid':0.8412},
    {'Date':'2024-07-24','Open':0.8409,'Close':0.8399,'High':0.8418,'Low':0.8394,'Mid':0.8406},
    {'Date':'2024-07-25','Open':0.8399,'Close':0.8439,'High':0.8440,'Low':0.8396,'Mid':0.8418},
    {'Date':'2024-07-26','Open':0.8439,'Close':0.8437,'High':0.8449,'Low':0.8428,'Mid':0.8439},
    {'Date':'2024-07-29','Open':0.8447,'Close':0.8414,'High':0.8461,'Low':0.8412,'Mid':0.8436},
    {'Date':'2024-07-30','Open':0.8414,'Close':0.8426,'High':0.8430,'Low':0.8411,'Mid':0.8420},
    {'Date':'2024-07-31','Open':0.8426,'Close':0.8421,'High':0.8449,'Low':0.8417,'Mid':0.8433},

    # ===================== AUG 2024 =====================
    {'Date':'2024-08-01','Open':0.8421,'Close':0.8471,'High':0.8473,'Low':0.8419,'Mid':0.8446},
    {'Date':'2024-08-02','Open':0.8471,'Close':0.8522,'High':0.8535,'Low':0.8467,'Mid':0.8501},
    {'Date':'2024-08-05','Open':0.8531,'Close':0.8571,'High':0.8619,'Low':0.8524,'Mid':0.8572},
    {'Date':'2024-08-06','Open':0.8571,'Close':0.8613,'High':0.8616,'Low':0.8562,'Mid':0.8589},
    {'Date':'2024-08-07','Open':0.8613,'Close':0.8605,'High':0.8615,'Low':0.8576,'Mid':0.8596},
    {'Date':'2024-08-08','Open':0.8606,'Close':0.8564,'High':0.8625,'Low':0.8560,'Mid':0.8592},
    {'Date':'2024-08-09','Open':0.8564,'Close':0.8557,'High':0.8579,'Low':0.8550,'Mid':0.8564},
    {'Date':'2024-08-12','Open':0.8548,'Close':0.8561,'High':0.8569,'Low':0.8544,'Mid':0.8556},
    {'Date':'2024-08-13','Open':0.8561,'Close':0.8547,'High':0.8566,'Low':0.8531,'Mid':0.8549},
    {'Date':'2024-08-14','Open':0.8547,'Close':0.8585,'High':0.8593,'Low':0.8541,'Mid':0.8567},
    {'Date':'2024-08-15','Open':0.8585,'Close':0.8536,'High':0.8587,'Low':0.8534,'Mid':0.8561},
    {'Date':'2024-08-16','Open':0.8536,'Close':0.8520,'High':0.8538,'Low':0.8511,'Mid':0.8524},
    {'Date':'2024-08-19','Open':0.8530,'Close':0.8532,'High':0.8535,'Low':0.8508,'Mid':0.8522},
    {'Date':'2024-08-20','Open':0.8532,'Close':0.8539,'High':0.8540,'Low':0.8515,'Mid':0.8528},
    {'Date':'2024-08-21','Open':0.8539,'Close':0.8520,'High':0.8545,'Low':0.8510,'Mid':0.8527},
    {'Date':'2024-08-22','Open':0.8520,'Close':0.8488,'High':0.8523,'Low':0.8479,'Mid':0.8501},
    {'Date':'2024-08-23','Open':0.8489,'Close':0.8468,'High':0.8493,'Low':0.8451,'Mid':0.8472},
    {'Date':'2024-08-26','Open':0.8475,'Close':0.8464,'High':0.8476,'Low':0.8450,'Mid':0.8463},
    {'Date':'2024-08-27','Open':0.8464,'Close':0.8434,'High':0.8468,'Low':0.8432,'Mid':0.8450},
    {'Date':'2024-08-28','Open':0.8435,'Close':0.8430,'High':0.8438,'Low':0.8410,'Mid':0.8424},
    {'Date':'2024-08-29','Open':0.8430,'Close':0.8413,'High':0.8434,'Low':0.8403,'Mid':0.8418},
    {'Date':'2024-08-30','Open':0.8413,'Close':0.8414,'High':0.8428,'Low':0.8400,'Mid':0.8414},

    # ===================== SEP 2024 =====================
    {'Date':'2024-09-02','Open':0.8413,'Close':0.8422,'High':0.8433,'Low':0.8411,'Mid':0.8422},
    {'Date':'2024-09-03','Open':0.8422,'Close':0.8421,'High':0.8435,'Low':0.8406,'Mid':0.8421},
    {'Date':'2024-09-04','Open':0.8422,'Close':0.8432,'High':0.8435,'Low':0.8416,'Mid':0.8426},
    {'Date':'2024-09-05','Open':0.8432,'Close':0.8431,'High':0.8438,'Low':0.8418,'Mid':0.8428},
    {'Date':'2024-09-06','Open':0.8431,'Close':0.8442,'High':0.8447,'Low':0.8412,'Mid':0.8429},
    {'Date':'2024-09-09','Open':0.8445,'Close':0.8441,'High':0.8448,'Low':0.8433,'Mid':0.8440},
    {'Date':'2024-09-10','Open':0.8441,'Close':0.8424,'High':0.8451,'Low':0.8424,'Mid':0.8437},
    {'Date':'2024-09-11','Open':0.8424,'Close':0.8443,'High':0.8464,'Low':0.8423,'Mid':0.8443},
    {'Date':'2024-09-12','Open':0.8443,'Close':0.8438,'High':0.8455,'Low':0.8434,'Mid':0.8445},
    {'Date':'2024-09-13','Open':0.8438,'Close':0.8438,'High':0.8453,'Low':0.8427,'Mid':0.8440},
    {'Date':'2024-09-16','Open':0.8441,'Close':0.8424,'High':0.8444,'Low':0.8420,'Mid':0.8432},
    {'Date':'2024-09-17','Open':0.8424,'Close':0.8445,'High':0.8454,'Low':0.8419,'Mid':0.8436},
    {'Date':'2024-09-18','Open':0.8445,'Close':0.8415,'High':0.8453,'Low':0.8404,'Mid':0.8429},
    {'Date':'2024-09-19','Open':0.8415,'Close':0.8403,'High':0.8423,'Low':0.8392,'Mid':0.8408},
    {'Date':'2024-09-20','Open':0.8403,'Close':0.8380,'High':0.8407,'Low':0.8378,'Mid':0.8392},
    {'Date':'2024-09-23','Open':0.8384,'Close':0.8325,'High':0.8387,'Low':0.8323,'Mid':0.8355},
    {'Date':'2024-09-24','Open':0.8325,'Close':0.8335,'High':0.8340,'Low':0.8317,'Mid':0.8328},
    {'Date':'2024-09-25','Open':0.8335,'Close':0.8355,'High':0.8372,'Low':0.8334,'Mid':0.8353},
    {'Date':'2024-09-26','Open':0.8355,'Close':0.8332,'High':0.8361,'Low':0.8327,'Mid':0.8344},
    {'Date':'2024-09-27','Open':0.8332,'Close':0.8348,'High':0.8350,'Low':0.8322,'Mid':0.8336},
    {'Date':'2024-09-30','Open':0.8346,'Close':0.8325,'High':0.8360,'Low':0.8313,'Mid':0.8336},

    # ===================== OCT 2024 =====================
    {'Date':'2024-10-01','Open':0.8325,'Close':0.8331,'High':0.8345,'Low':0.8311,'Mid':0.8328},
    {'Date':'2024-10-02','Open':0.8331,'Close':0.8325,'High':0.8338,'Low':0.8322,'Mid':0.8330},
    {'Date':'2024-10-03','Open':0.8325,'Close':0.8404,'High':0.8434,'Low':0.8324,'Mid':0.8379},
    {'Date':'2024-10-04','Open':0.8404,'Close':0.8364,'High':0.8407,'Low':0.8353,'Mid':0.8380},
    {'Date':'2024-10-07','Open':0.8367,'Close':0.8388,'High':0.8400,'Low':0.8356,'Mid':0.8378},
    {'Date':'2024-10-08','Open':0.8388,'Close':0.8379,'High':0.8405,'Low':0.8374,'Mid':0.8389},
    {'Date':'2024-10-09','Open':0.8380,'Close':0.8369,'High':0.8388,'Low':0.8367,'Mid':0.8378},
    {'Date':'2024-10-10','Open':0.8369,'Close':0.8375,'High':0.8386,'Low':0.8355,'Mid':0.8370},
    {'Date':'2024-10-11','Open':0.8375,'Close':0.8370,'High':0.8383,'Low':0.8365,'Mid':0.8374},
    {'Date':'2024-10-14','Open':0.8364,'Close':0.8353,'High':0.8374,'Low':0.8349,'Mid':0.8361},
    {'Date':'2024-10-15','Open':0.8354,'Close':0.8331,'High':0.8356,'Low':0.8326,'Mid':0.8341},
    {'Date':'2024-10-16','Open':0.8331,'Close':0.8362,'High':0.8380,'Low':0.8327,'Mid':0.8354},
    {'Date':'2024-10-17','Open':0.8361,'Close':0.8324,'High':0.8370,'Low':0.8318,'Mid':0.8344},
    {'Date':'2024-10-18','Open':0.8324,'Close':0.8327,'High':0.8336,'Low':0.8295,'Mid':0.8316},
    {'Date':'2024-10-21','Open':0.8323,'Close':0.8329,'High':0.8340,'Low':0.8323,'Mid':0.8331},
    {'Date':'2024-10-22','Open':0.8329,'Close':0.8316,'High':0.8347,'Low':0.8314,'Mid':0.8331},
    {'Date':'2024-10-23','Open':0.8317,'Close':0.8344,'High':0.8345,'Low':0.8304,'Mid':0.8324},
    {'Date':'2024-10-24','Open':0.8344,'Close':0.8344,'High':0.8352,'Low':0.8315,'Mid':0.8333},
    {'Date':'2024-10-25','Open':0.8345,'Close':0.8332,'High':0.8350,'Low':0.8326,'Mid':0.8338},
    {'Date':'2024-10-28','Open':0.8325,'Close':0.8335,'High':0.8344,'Low':0.8322,'Mid':0.8333},
    {'Date':'2024-10-29','Open':0.8335,'Close':0.8312,'High':0.8341,'Low':0.8299,'Mid':0.8320},
    {'Date':'2024-10-30','Open':0.8312,'Close':0.8375,'High':0.8377,'Low':0.8307,'Mid':0.8342},
    {'Date':'2024-10-31','Open':0.8375,'Close':0.8438,'High':0.8448,'Low':0.8354,'Mid':0.8401},

    # ===================== NOV 2024 =====================
    {'Date':'2024-11-01','Open':0.8438,'Close':0.8382,'High':0.8443,'Low':0.8369,'Mid':0.8406},
    {'Date':'2024-11-04','Open':0.8389,'Close':0.8395,'High':0.8419,'Low':0.8385,'Mid':0.8402},
    {'Date':'2024-11-05','Open':0.8396,'Close':0.8381,'High':0.8403,'Low':0.8378,'Mid':0.8391},
    {'Date':'2024-11-06','Open':0.8380,'Close':0.8331,'High':0.8386,'Low':0.8315,'Mid':0.8350},
    {'Date':'2024-11-07','Open':0.8330,'Close':0.8319,'High':0.8341,'Low':0.8306,'Mid':0.8323},
    {'Date':'2024-11-08','Open':0.8319,'Close':0.8297,'High':0.8326,'Low':0.8292,'Mid':0.8309},
    {'Date':'2024-11-11','Open':0.8298,'Close':0.8281,'High':0.8303,'Low':0.8260,'Mid':0.8281},
    {'Date':'2024-11-12','Open':0.8281,'Close':0.8333,'High':0.8335,'Low':0.8276,'Mid':0.8306},
    {'Date':'2024-11-13','Open':0.8334,'Close':0.8313,'High':0.8346,'Low':0.8308,'Mid':0.8327},
    {'Date':'2024-11-14','Open':0.8313,'Close':0.8313,'High':0.8379,'Low':0.8307,'Mid':0.8343},
    {'Date':'2024-11-15','Open':0.8313,'Close':0.8352,'High':0.8359,'Low':0.8311,'Mid':0.8335},
    {'Date':'2024-11-18','Open':0.8338,'Close':0.8360,'High':0.8373,'Low':0.8337,'Mid':0.8355},
    {'Date':'2024-11-19','Open':0.8360,'Close':0.8355,'High':0.8375,'Low':0.8332,'Mid':0.8354},
    {'Date':'2024-11-20','Open':0.8355,'Close':0.8333,'High':0.8359,'Low':0.8313,'Mid':0.8336},
    {'Date':'2024-11-21','Open':0.8333,'Close':0.8320,'High':0.8341,'Low':0.8317,'Mid':0.8329},
    {'Date':'2024-11-22','Open':0.8320,'Close':0.8316,'High':0.8347,'Low':0.8267,'Mid':0.8307},
    {'Date':'2024-11-25','Open':0.8329,'Close':0.8351,'High':0.8361,'Low':0.8309,'Mid':0.8335},
    {'Date':'2024-11-26','Open':0.8351,'Close':0.8346,'High':0.8365,'Low':0.8331,'Mid':0.8348},
    {'Date':'2024-11-27','Open':0.8346,'Close':0.8333,'High':0.8355,'Low':0.8332,'Mid':0.8343},
    {'Date':'2024-11-28','Open':0.8333,'Close':0.8319,'High':0.8339,'Low':0.8315,'Mid':0.8327},
    {'Date':'2024-11-29','Open':0.8319,'Close':0.8305,'High':0.8332,'Low':0.8302,'Mid':0.8317},

    # ===================== DEC 2024 (partial, up to Dec 26) =====================
    {'Date':'2024-12-02','Open':0.8308,'Close':0.8296,'High':0.8308,'Low':0.8270,'Mid':0.8289},
    {'Date':'2024-12-03','Open':0.8296,'Close':0.8293,'High':0.8313,'Low':0.8287,'Mid':0.8300},
    {'Date':'2024-12-04','Open':0.8293,'Close':0.8276,'High':0.8302,'Low':0.8269,'Mid':0.8285},
    {'Date':'2024-12-05','Open':0.8276,'Close':0.8297,'High':0.8301,'Low':0.8273,'Mid':0.8287},
    {'Date':'2024-12-06','Open':0.8297,'Close':0.8294,'High':0.8301,'Low':0.8279,'Mid':0.8290},
    {'Date':'2024-12-09','Open':0.8293,'Close':0.8277,'High':0.8293,'Low':0.8269,'Mid':0.8281},
    {'Date':'2024-12-10','Open':0.8277,'Close':0.8243,'High':0.8283,'Low':0.8239,'Mid':0.8261},
    {'Date':'2024-12-11','Open':0.8243,'Close':0.8232,'High':0.8252,'Low':0.8225,'Mid':0.8239},
    {'Date':'2024-12-12','Open':0.8232,'Close':0.8259,'High':0.8273,'Low':0.8227,'Mid':0.8250},
    {'Date':'2024-12-13','Open':0.8259,'Close':0.8323,'High':0.8324,'Low':0.8255,'Mid':0.8290},
    {'Date':'2024-12-16','Open':0.8317,'Close':0.8288,'High':0.8328,'Low':0.8273,'Mid':0.8301},
    {'Date':'2024-12-17','Open':0.8288,'Close':0.8254,'High':0.8296,'Low':0.8250,'Mid':0.8273},
    {'Date':'2024-12-18','Open':0.8254,'Close':0.8234,'High':0.8278,'Low':0.8230,'Mid':0.8254},
    {'Date':'2024-12-19','Open':0.8234,'Close':0.8289,'High':0.8293,'Low':0.8223,'Mid':0.8258},
    {'Date':'2024-12-20','Open':0.8289,'Close':0.8299,'High':0.8313,'Low':0.8271,'Mid':0.8292},
    {'Date':'2024-12-23','Open':0.8303,'Close':0.8301,'High':0.8316,'Low':0.8278,'Mid':0.8297},
    {'Date':'2024-12-24','Open':0.8301,'Close':0.8293,'High':0.8303,'Low':0.8275,'Mid':0.8289},
    {'Date':'2024-12-26','Open':0.8293,'Close':0.8321,'High':0.8325,'Low':0.8287,'Mid':0.8306},
]

############################################################################
# 2) PREP THE DATA
############################################################################

def load_full_2024_data():
    """
    Create a DataFrame from the entire train_data array (Jan 1 -> Dec 26).
    """
    df = pd.DataFrame(train_data)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.drop(columns=['Mid'], inplace=True)   # we don't need 'Mid'
    df.sort_index(inplace=True)
    return df

def clean_data(df):
    """
    Simple cleaning: sort, remove duplicates, drop NaN if any exist.
    """
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df.dropna(inplace=True)
    return df

############################################################################
# 3) BUILD DATASET (multi-window, single-step)
############################################################################

def build_dataset(df, w_max=5):
    """
    We'll do single-step next-day forecasting:
      X = flattened last w days of O,H,L,C
      Y = next day's [Low, High, Close]
    We'll gather multiple windows: w=1..w_max
    Return X, Y, plus meta_info for reference.
    """
    df_np = df[['Open','High','Low','Close']].values
    dates = df.index.to_list()
    n = len(df_np)

    # We'll store all training samples in these lists.
    X_list, Y_list, info_list = [], [], []

    max_len = w_max * 4  # For w rows, each has 4 columns => O,H,L,C

    # For each w in [1..w_max], for each t in [w..(n-1)]:
    for w in range(1, w_max+1):
        for t in range(w, n):  
            # We'll predict the day t's [Low,High,Close] from the prior w days.
            # But if t == n => that would have no "next day" in real usage,
            # though we could do it for a final partial. Usually we do t in [w..n-1].
            if t >= n: 
                break

            # Flatten the last w days into one vector
            window_data = df_np[t-w : t, :]  # shape (w,4)
            window_flat = window_data.flatten()  # shape (w*4,)

            # Pad to a length = max_len if w < w_max
            padded = np.zeros(max_len, dtype=float)
            start_idx = max_len - len(window_flat)
            padded[start_idx:] = window_flat

            # Next-day's O,H,L,C is day t
            # We'll only predict [Low,High,Close], ignoring 'Open' in Y.
            future_day = df_np[t]   # shape(4,)
            day_low   = future_day[2]
            day_high  = future_day[1]
            day_close = future_day[3]

            # Our target
            target = np.array([day_low, day_high, day_close], dtype=float)

            # Optionally embed w as a feature
            feature = np.concatenate([padded, [float(w)]], axis=0)

            X_list.append(feature)
            Y_list.append(target)
            info_list.append((w, dates[t]))  # which w & which date are we predicting

    X = np.array(X_list, dtype=float)
    Y = np.array(Y_list, dtype=float)
    return X, Y, info_list

############################################################################
# 4) TRAIN MODEL
############################################################################

def train_model(X, Y):
    """
    Train a multi-output regressor (LinearRegression).
    """
    base = LinearRegression()
    model = MultiOutputRegressor(base)
    model.fit(X, Y)
    return model

############################################################################
# 5) PREDICT JUST THE NEXT DAY: 2024-12-27
############################################################################

def predict_next_trading_day(df, model, w_max=5):
    """
    We'll define the 'latest day' in df as 2024-12-26 (the last row).
    We gather the last w_max days, flatten, predict day (12-27).
    """
    # We only want the last w_max rows
    recent_df = df[['Open','High','Low','Close']].tail(w_max)
    arr = recent_df.values.flatten()

    max_len = w_max * 4
    padded = np.zeros(max_len, dtype=float)
    start_idx = max_len - len(arr)
    padded[start_idx:] = arr

    # embed w as well
    final_feature = np.concatenate([padded, [float(w_max)]], axis=0).reshape(1, -1)

    pred = model.predict(final_feature)[0]  # shape (3,)
    pred_low, pred_high, pred_close = pred
    # We'll do a naive "Open = prior day's close" or skip it
    # For a single day, we'll just return the [Low,High,Close]
    return pred_low, pred_high, pred_close

############################################################################
# 6) MAIN
############################################################################

def main():
    df_2024 = load_full_2024_data()
    df_2024 = clean_data(df_2024)

    # We'll exclude the final row (Dec 26) from training, if we want 
    # truly out-of-sample. Or we can use all data. 
    # Let's train on everything up through Dec 26 - 1 => Dec 24? 
    # But let's keep it simpler & train on everything except the very last row.
    # So final row index is for Dec 26:
    final_index = df_2024.index[-1]   # 2024-12-26
    # We'll train on everything up to (but not including) the last row
    df_train = df_2024.iloc[:-1].copy()
    print(f"Training on {len(df_train)} rows. Final row excluded: {final_index}")

    # Build dataset
    X, Y, info = build_dataset(df_train, w_max=5)
    print(f"Dataset shapes: X={X.shape}, Y={Y.shape}")
    print(f"Sample info last 5:\n{info[-5:]}")

    # Train
    model = train_model(X, Y)
    print("Model training complete.")

    # Predict next day => that day is 2024-12-27
    pred_low, pred_high, pred_close = predict_next_trading_day(df_2024, model, w_max=5)

    # Print result
    print("\n=== PREDICTION FOR 2024-12-27 ===")
    print(f"Predicted Low:   {pred_low:.5f}")
    print(f"Predicted High:  {pred_high:.5f}")
    print(f"Predicted Close: {pred_close:.5f}")

if __name__ == "__main__":
    main()
