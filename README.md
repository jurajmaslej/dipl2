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
- kostra práce v laTex v repozitári

### Implementácia
- načítanie .tif, prevod do jpeg
- práca s HSV formátom
- vytvorenie masky na filtráciu oblohy
- trénovanie kernelu konvolučnej siete

### Postup implementácie
 Loader súbory: load_camera, load_synops, main_loader
 Načítavajú fisheye snímky a synopy
 Pomocné súbory:
 helper - metódy zdieľané medzi viacerými súbormi
 histogram_comparer - porovnávanie histogramov
 concepts - trash folder pre nápady
 
### Úprava vstupných dát

#### Synopy
K dispozícií máme meteorologické dátat z 2 lokalít, z nich dostávame synopy.
Výsledkom su 2 slovníky, kľúčom je dátum, hodnota je synopa.
Pridali sme tretí, ktorý je priemerom synop.
#### Fotky
Fotografie sú z kamery s rybím okom. Zmenšujeme ich na rozmer 150x150.
Súbor fotografií je filtrovaný na denné fotografie.
V súbore sa tiež nachádzalo značné množstvo fotografii, kde bola kamera nevyhovujúco otočená (zem vs. obloha)
Tieto fotografie odstraňujeme pomocou porovnávanie histogramov.
Prevádzame na HSV formát. Následne vytvoríme slovník kde kľúčom je dátum s rovnakom formate ako pri synopách, hodnotou je fotografia načítaná pomocou openCV.
Ďalej obdobný slovník vytvárame s histogrammi fotografíí už prevdných do HSV spektra.

### Plánované postupy
Pracujeme na použití vhodného machine learning modelu na kategorizáiu oblačnosti na fotkách vzhľadom k ich histogramom. Toto riešenie má slúžiť ako baseline pre následné vystavanie modelu konvolučnej neurónovej siete

### Literatúra
 Ryo Onishi, Deep convolutional neural network for cloud estimation from snapshot camera images, 2017<br>
 Le Goff, Deep learning for cloud detection, 2017<br>
 Zafarifar, Weda, Horizon detection, 2008<br>
 Goodfellow, Deep Learning. 2016
 - odkazy v dokumente prezent.pdf

## V súčasnosti dáta nie sú na githube (pamäťové obmedzenie)
