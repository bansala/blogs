#Feature Engineering with Dates - Part 1

>_**Feature engineering is often the most important part…**_

is what research on winning Kaggle’rs by [David Wind reveals](http://blog.kaggle.com/2014/08/01/learning-from-the-best/). It is somewhat ignored in the texts about machine learning, which seem to focus more on the algorithms, rather than what is being fed into these algorithms. What is feature engineering then?

>_**Feature engineering is the process of transforming raw data into features that better represent the underlying problem to the predictive models, resulting in improved model accuracy on unseen data.**_ [^1]

The focus of this article will be on a simple date/time stamp field and see what features could be extracted from this one field. 

##Data 
To illustrate what is possible, we will consider a simple transaction data set, one possibly generated from retail purchases. Let’s say that we have simple transaction table, with a column identifying the customer, a column indicating the product that was purchased, a column for price and column containing the date and time the purchase was made. Lets say this data is available in a CSV. A file could look like this:

customer id	| product id	|price	|date time stamp|
----------	|--------------	|-------|---------------
1			| 5	| 5.99	| 2015–09–11 11:23
1	| 3	| 2.99	| 2015–01–12 17:50

You could obtain such a data set from [Kaggle’s Acquire Valued Shopper Challenge](https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data). Look for the transactions data. Note that in this data set, there is no price. 

For this example, we will be using the [Fa-Teng data set](http://www.bigdatalab.ac.cn/benchmark/bm/dd?data=Ta-Feng). There are a number of other data sets for grocery/retail in [Recsys](http://recsyswiki.com/wiki/Grocery_shopping_datasets).

>*Side Note*: For a lot of the analysis and code, I will be using Python. I use the [Pineapple/Jupyter](https://nwhitehead.github.io/pineapple/) IPython notebook. 

## Round 1: Basic Features
When we look at a date time stamp, a number of features, or pieces of information are immediately obvious:

* Year
* Month
* Day
* Day of week
* Week of year
* Hour of day

Month and day of week can be quite useful in understanding periodicity or seasonality of transactions. We may find that some actions are more probable on certain days of the week, or somethings happen around the same month every year. With [Halloween](https://en.wikipedia.org/wiki/Halloween) around the corner for example, you are probably shopping for candy right now. 

Using `pandas` we try and load this data set (you may have to remove the header row from the file):

```

columns = ['date', 'customer', 'age', 'zipcode', 'product_class', 'product_id', 'amount', 'asset', 'price']

txs = pd.read_table('D11-02/D01', sep=';', header=None, names=columns)
txs.info()  # to get summary statistics
txs.head()  # to get a feel for the data

```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>customer</th>
      <th>age</th>
      <th>zipcode</th>
      <th>product_class</th>
      <th>product_id</th>
      <th>amount</th>
      <th>asset</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001-01-01 00:00:00</td>
      <td>141833</td>
      <td>F</td>
      <td>F</td>
      <td>130207</td>
      <td>4710105011011</td>
      <td>2</td>
      <td>44</td>
      <td>52</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001-01-01 00:00:00</td>
      <td>1376753</td>
      <td>E</td>
      <td>E</td>
      <td>110217</td>
      <td>4710265849066</td>
      <td>1</td>
      <td>150</td>
      <td>129</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2001-01-01 00:00:00</td>
      <td>1603071</td>
      <td>E</td>
      <td>G</td>
      <td>100201</td>
      <td>4712019100607</td>
      <td>1</td>
      <td>35</td>
      <td>39</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001-01-01 00:00:00</td>
      <td>1738667</td>
      <td>E</td>
      <td>F</td>
      <td>530105</td>
      <td>4710168702901</td>
      <td>1</td>
      <td>94</td>
      <td>119</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2001-01-01 00:00:00</td>
      <td>2141497</td>
      <td>A</td>
      <td>B</td>
      <td>320407</td>
      <td>4710431339148</td>
      <td>1</td>
      <td>100</td>
      <td>159</td>
    </tr>
  </tbody>
</table>
</div>

Unfortunately, the time stamps in this dataset are useless. I couldn't find a realistic data set which has time stamp information. Welcome to the real world, with imperfect data! However, if you know of a good data set, I would love to hear from you!

For the purpose of our feature engineering, lets just imagine that time stamps are available. Now, lets start adding our first set of features to this data set.

```
from datetime import datetime

year = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S" ).year
txs['year'] = txs['date'].map(year)
txs.head()

```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>customer</th>
      <th>age</th>
      <th>zipcode</th>
      <th>product_class</th>
      <th>product_id</th>
      <th>amount</th>
      <th>asset</th>
      <th>price</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001-01-01 00:00:00</td>
      <td>141833</td>
      <td>F</td>
      <td>F</td>
      <td>130207</td>
      <td>4710105011011</td>
      <td>2</td>
      <td>44</td>
      <td>52</td>
      <td>2001</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001-01-01 00:00:00</td>
      <td>1376753</td>
      <td>E</td>
      <td>E</td>
      <td>110217</td>
      <td>4710265849066</td>
      <td>1</td>
      <td>150</td>
      <td>129</td>
      <td>2001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2001-01-01 00:00:00</td>
      <td>1603071</td>
      <td>E</td>
      <td>G</td>
      <td>100201</td>
      <td>4712019100607</td>
      <td>1</td>
      <td>35</td>
      <td>39</td>
      <td>2001</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001-01-01 00:00:00</td>
      <td>1738667</td>
      <td>E</td>
      <td>F</td>
      <td>530105</td>
      <td>4710168702901</td>
      <td>1</td>
      <td>94</td>
      <td>119</td>
      <td>2001</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2001-01-01 00:00:00</td>
      <td>2141497</td>
      <td>A</td>
      <td>B</td>
      <td>320407</td>
      <td>4710431339148</td>
      <td>1</td>
      <td>100</td>
      <td>159</td>
      <td>2001</td>
    </tr>
  </tbody>
</table>
</div>
You can see here that the feature was added to the `DataFrame`.
Here are some other map functions you could use:

```
day_of_week = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S" ).weekday()
month = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S" ).month
# please read docs on how week numbers are calculate
week_number = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S" ).strftime('%V')
```

You can try writing some of the other features we mentioned above yourself. See, with such simple code, we just added 7 new features!

## Round 2: More Interesting Features
Now, lets think of more interesting features that may involve looks. How about seasons, or times of days? Here are some example maps that you could run:

```
seasons = [0,0,1,1,1,2,2,2,3,3,3,0]  #dec - feb is winter, then spring, summer, fall etc
season = lambda x: seasons[(datetime.strptime(x, "%Y-%m-%d %H:%M:%S" ).month-1)]

# sleep: 12-5, 6-9: breakfast, 10-14: lunch, 14-17: dinner prep, 17-21: dinner, 21-23: deserts!
times_of_day = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5 ]
time_of_day = lambda x: times_of_day[datetime.strptime(x, "%Y-%m-%d %H:%M:%S").hour]
```

I was working for a client that had a ton of recipes on their site. One of the questions they asked was about a content recommender and what is the right time to display what type of recipe. We used this time of day map to understand when people see breakfast recipe, kids lunch box recipes, appetizers etc. Intuitively, you can imagine that people prepare for next day's lunch and breakfast around or after dinner, especially if you have children. If you coming to the end of your work day, you are probably thinking about dinner and what you could pick up on you way home. This feature was extracted from clickstream to enrich the data and give additional insight in to what types of recipes to show at what time. Season was also a good predictor to understand which recipes are timeless and which are more seasonal. For another grocery client, we saw huge uptick in browsing flyers between 8 and 10 am and noon to 1pm (lunch hour) during week days. Further, Wednesday and Thursdays were heaviest traffic days of the week as most people planned for grocery shopping just before the weekend.

## The Next Part
So far, we have added 9 features from one date time column. This is by no means an exhaustive list, nor are we done. There are some interesting features that are still waiting to be teased out. In the next article in this series, I will cover extracting more features by merging external data like integrating weather or public holiday data, and maybe do a little bit of calculus! 

In the meanwhile, try these out, and let me know how these work for you!

[^1]: http://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/
