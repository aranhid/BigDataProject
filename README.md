# Генерация мелодии при помощи LSTM нейросети на основе midi файлов
# Авторы: Ковальчук Дмитрий и Цыбуля Владислав
## Цель
Сгенерировать мелодию при помощи LSTM нейросети, обученной на на нескольких midi файлах.
## Задачи
1. Собрать данные для обучения.
2. Конвертировать исходные данные в формат, который сможет прочитать нейросеть.
3. Создать архитектуру и обучить LSTM нейросеть.
4. По обученной модели сгенерировать мелодию.
Сравнить результаты на разных конфигурациях обучения и сделать выводы.
## Алгоритм решения
Программа состоит из двух скриптов: [learning.py](learning.py) и [generate.py](generate.py).
### Обучение
Скрипт [learning.py](learning.py) читает midi файлы из заданной директории, с помощью библиотеки music21 разбивает каждый файл на последовательности нот с их длительностями, и потом с помощью one-hot encoding подготавливается массив вида [(Нота, длительность), (Нота, длительность)...] . После этого подготовленный массив передается в LSTM нейросеть для её обучения. После обученная модель сохраняется в файл.
### Генерация
Скрипт [generate.py](generate.py) считывыет midi файлы из директории, используя библиотеку music21 делает массив всех используемых нот и их длительностей. Затем загружается обученная модель нейросети, и с её помощью генерируются данные для музыкального файла. 

Генерация происходит следующим образом: берется последовательность из n нот, на её основе генерируется новая последовательность из n нот, и потом на основе только что сгенерированной последовательности генерируется новая последовательность. 

После данные, сгенерированные нейросетью, приводятся к виду, в каком они были получены из midi файла. Ноты расставляются по своим местам, им присваивается нужная длительность, и всё это записывается в выходной midi файл.
## Исходные данные для обучения
В качестве данных для обучения взяты классические произведения Бетховена и Моцарта в файлах формата midi, так как этот формат удобен для преобразования данных в вид, пригодный для нейросети. Файлы формата midi хорошо преобразуются в последовательности нот и их длительностей, и именно на этих данных обучается нейросеть.

[Датасет 1: classic](classic)

[Датасет 2: midi](midi)
## Пример работы программы
Пример при обучении на 5 файлах из папки midi, с длинной последовательности нот равной 100 и 10 эпохами обучения: [midi_Model_Seq100_Files5_Epoch10.mid](output/midi_Model_Seq100_Files5_Epoch10.mid)

Пример при обучении на 10 файлах из папки classic, с длинной последовательности нот равной 10 и 10 эпохами обучения: [classic_Model_Seq10_Files10_Epoch10.mid](output/classic_Model_Seq10_Files10_Epoch10.mid)

Пример при обучении на 8 файлах из папки classic, с длинной последовательности нот равной 50 и 10 эпохами обучения: [classic_Model_Seq50_Files8_Epoch10.mid](output/classic_Model_Seq50_Files8_Epoch10.mid)
## Выводы и дальнейшие планы
Чем большую последовательность нот мы подаем для обучения нейросети, тем более интересная получается мелодия. Однако, значительно возрастает время обучения, в следствии чего становится целесообразным обучать нейросеть на GPU. Кроме того, мы использовали для обучения 10 эпох, но этого недостаточно для получения удовлетворительного результата.

Дальнейшие планы:
* Протестировать программу на большем количестве эпох.
* Найти оптимальное количество эпох и длину последовательности нот.

## Необходимые библиотеки:
> pip3 install tensorflow==1.14.0 Keras==2.3.1 music21

## Запуск:
### Обучение:
> python3 learning.py
### Генерация
> python3 generate.py