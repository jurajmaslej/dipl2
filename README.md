# Diplomová práca

## Možnosti využitia metód hlbokého učenia v predpovedi počasia

Kompilačná a čiastočne implementačná práca zo strojového učenia
s fyzikálnym rozmerom

### Cieľ

Hlboké učenie využíva konvolučné neurónové siete, kde sa z dát učí tzv. kernel,
ktorý sa paralelne aplikuje na na všetky miesta na dátach a spracuje lokálne
okolie tohto miesta na zložku nadradenej dátovej úrovne. Táto metóda sa
s úspechom používa na spracovanie obrazu. Cieľom tejto práce je preskúmať
možnosti jej aplikácie v inej doméne a to pri spracovaní meteorologických
údajov. Tieto údaje majú tiež 2D charakter ako obraz, avšak rôzne zložky
v rôznych jednotkách: jednotlivé meteorologické veličiny (teplota, vlhkosť,
tlak, rýchlosť vetra) a geografické dáta (nadmorská výška, zemepisná dĺžka, ...).
Údaje s potrebnou anotáciou budú poskytnuté. Počítačový jazyk a knižnica
pre hlboké učenie si diplomant vyberie sám, avšak použije existujúce riešenie,
odporučané prostredie je tensorflow. Hardwarová platforma na rozsiahle
výpočty potrebné pre spracovanie dát, bude poskytnuté.

### Súčasný stav
- výber technológii ukončený 
- prvé obrazové dáta
- práca s HSV formátom
- práca s opencv a keras
- prehľad v literatúre
- prvá kapitola v repozitári 

### Filtrácia dát
- na základe časových údajov, nočné snímky pre nás nemajú hodnotu
- niektoré snímky neobsahujú oblohu ale výlučne zemský povrch, filtrácia na základe histogramu
- porovnávanie snímok s ručne vybraným setom snímok na určenie či sa jedná o záber na oblohu alebo nie
- úprava fotiek na rozmer 150x150
- úprava do HSV formátu

### Implementácia
- načítanie .tif, prevod do jpeg
- práca s HSV formátom
- vytvorenie masky na filtráciu oblohy
- vyskúšané na malých dátach trénovanie kernelu konvolučnej siete
- klasifikácia oblačnosti pomocou náhodného stromu

### Literatúra
 Ryo Onishi, Deep convolutional neural network for cloud estimation from snapshot camera images, 2017<br>
 Le Goff, Deep learning for cloud detection, 2017<br>
 Zafarifar, Weda, Horizon detection, 2008<br>
 Goodfellow, Deep Learning. 2016
 - odkazy v dokumente prezent.pdf

## V súčasnosti dáta nie sú na githube (pamäťové obmedzenie)
