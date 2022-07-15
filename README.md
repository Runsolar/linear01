# Модель персептрона

## Общие сведения
*Персептрон (нейрон Розенблатта)* - одна из первых моделей нейросетей предложенная Фрэнком Розенблаттом в 1958 году.
На вход эта модель принимает вектор входных параметров Х = (x1, … , xn) и некоторое смещение bias. Каждому компоненту вектора входных параметров xi соответствует весовой коэффициент wi , определяющий величину воздействия компонента xi на персептрон.

Суммарное воздействие определяется по следующей формуле:

z = XW + b, где 

X - вектор входных параметров,<br/>
W - вектор весовых коэффициентов (в случае нескольких нейронов в модели - матрица),<br/>
b - смещение модели.

Для случая двух входных воздействий формула принимает следующий вид:

z = x1 * w1 + x2 * w2 + bias

После подсчета суммарного воздействия результат подается в функцию активации, результатом выполнения которой является предсказанное значение классификации y^, т.е. отнесение объекта к одному из двух предполагаемых классов.
В математическом представлении эта функция имеет вид:<br/>
y^(z) = 1, z>=0<br/>
y^(z) = -1, z<0

Так как мы имеем набор данных известное значение классификации, можем вычислить значение ошибки классификации персептрона:<br/>
delta = y - y^

С помощью этого значения мы можем обновить весовые коэффициенты модели и bias:<br/>
w1 = w1 + eta * (y - y^) * x1 <br/>
w2 = w2 + eta * (y - y^) * x2

где w1, w2 - весовые коэффициенты, соответствующие входным параметрам x1 и x2;<br/>
y - известное правильное значение классификации;<br/>
y^ - значение классификации, выданное нейроном;<br/>
eta - коэффициент скорости обучения, вычисляемый эмпирическим путем.<br/>
Данное соотношение мы можем использовать, если знаем начальные значения весовых коэффициентов w1 и w2. Единого стандарта для них нет, поэтому начальные значения задаются случайным числом от 0 до 1.

Таким образом, можем показать модель персептрона для случая двух входных параметров:    
![perceptron_picture](/imgs/перцептрон.jpg)

Итак, с помощью модели персептрона мы можем составить модель, которая будет определять принадлежность объекта к одному из классов. Однако стоит отметить, что эта модель будет работать верно *только при условии линейной разделимости классов*, т.е. при существовании гиперплоскости, отделяющей одно множество от другого.

## Сведения о программе
В программе присутствует классификатор: `class Perceptron(object)` с параметрами:<br/>
- `n_iter` - количество эпох
- `eta` - темп обучения
- `random_state` - параметр генератора случайных чисел
- `w` - одномерный массив весовых коэффициентов
- `errors` - одномерный массив ошибок классификации


**Методы класса:**
- `def fit(self, X, y)` - работа с моделью: обновление массива весовых коэффициентов, заполнение массива ошибок в каждой эпохе, где **X** - вектор тренирововчных занчений, **y** - целевые значения классификации;
- `def net_input(self, x)` - функция чистого входа (см. Общие сведения, функция **Z**), где **x** - вектор тренирововчных занчений;
- `def predict(self, x):` - функция классификации, где **x** - вектор тренирововчных занчений.

Функция разделения данных на тестовую и тренировочную выборки:<br/> `def train_test_split(x_input, y_input, test_percent, mixing):` ,<br/>
где `x_input` - вектор тренировочных значений, <br/>
`y_input` - целевые значения классификации, <br/>
`test_percent` - процент тестовой выборки [0, 1],
`mixing` - перемешивать ли выборки (True/False).<br/>
Возвращает функция четыре массива np.array(), соответствующие **X** тренировочный, **y** тренировочный, **X** тестовый, **y**  тестовый.

**Далее в программе происходит** 
- загрузка обучающей базы даныых:<br/> `irises = pd.read_csv('iris.csv', header=None, encoding='utf-8')` , <br/>`y = np.where(irises.iloc[1:100, -1] == 'Iris-setosa', 1, -1)`;
- загрузка тестовых значений и отделение от каждой группы тренировочной выборки:
    ```
    X = np.array(irises.iloc[1:100, [2, 4]]) 

    X_train_setosa = np.array(irises.iloc[1:36, [2, 4]])

    X_train_versicolor = np.array(irises.iloc[51:86, [2, 4]])

    X_train_virginica = np.array(irises.iloc[101:136, [2, 4]])

    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.3, True)
    ```
- обучение персептрона и вывод результатов прогноза и ошибок классификации:
    ```
    obj1  = Perceptron()

    obj1.fit(X_train, y_train)

    example = X_train

    print(obj1.predict(example))

    print(obj1.errors)
    ```
- вывод графика ошибок классификации в каждой эпохе:
    ```
    plt.plot(range(1, len(obj1.errors) + 1) , obj1.errors)
    plt.title('Количество ошибок классификации')
    plt.grid(True)
    plt.show()
    ```
## Структура файлов программы
![file_structure](/imgs/структура.png)

На картинке приведена структура программы, в центре которой находится исполняемый файл *classify.py*. По краям находятся файлы, которые передают обозначенные данные в исполняемый. Назначение и сожержание файлов приведены на схеме.

## Способ работы с программой

1. Скачать и распаковать репозиторий<br/>
![step1](/imgs/step1.png)

2. Открыть и запустить исполняемый файл программы **classify.py**<br/>

    ![step2](/imgs/step2.png)
    *пример запуска программы в среде разработки PyCharm*

## Пример результатов выполнеия программы
Вывод в консоль результатов классификации (1) и количества ошибок классификации в каждой эпохе (2):
![console_res](/imgs/console.png)

График количества ошибок классификации в каждой эпохе:
![err_gr](/imgs/err_gr.png)

Построение границы решений задачи классификации:
![bound_sol](/imgs/bound_sol.png)