# Q2 Part A: EDA Report

## Dataset Profile

- Raw shape: (891, 15)
- Cleaned shape: (889, 15)

### df.info()

```text
<class 'pandas.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 15 columns):
 #   Column       Non-Null Count  Dtype   
---  ------       --------------  -----   
 0   survived     891 non-null    int64   
 1   pclass       891 non-null    int64   
 2   sex          891 non-null    str     
 3   age          714 non-null    float64 
 4   sibsp        891 non-null    int64   
 5   parch        891 non-null    int64   
 6   fare         891 non-null    float64 
 7   embarked     889 non-null    str     
 8   class        891 non-null    category
 9   who          891 non-null    str     
 10  adult_male   891 non-null    bool    
 11  deck         203 non-null    category
 12  embark_town  889 non-null    str     
 13  alive        891 non-null    str     
 14  alone        891 non-null    bool    
dtypes: bool(2), category(2), float64(2), int64(4), str(5)
memory usage: 80.7 KB
```

### df.describe()

|             |   count |   unique | top         |   freq |       mean |        std |    min |      25% |      50% |   75% |     max |
|:------------|--------:|---------:|:------------|-------:|-----------:|-----------:|-------:|---------:|---------:|------:|--------:|
| survived    |     891 |      nan | nan         |    nan |   0.383838 |   0.486592 |   0    |   0      |   0      |     1 |   1     |
| pclass      |     891 |      nan | nan         |    nan |   2.30864  |   0.836071 |   1    |   2      |   3      |     3 |   3     |
| sex         |     891 |        2 | male        |    577 | nan        | nan        | nan    | nan      | nan      |   nan | nan     |
| age         |     714 |      nan | nan         |    nan |  29.6991   |  14.5265   |   0.42 |  20.125  |  28      |    38 |  80     |
| sibsp       |     891 |      nan | nan         |    nan |   0.523008 |   1.10274  |   0    |   0      |   0      |     1 |   8     |
| parch       |     891 |      nan | nan         |    nan |   0.381594 |   0.806057 |   0    |   0      |   0      |     0 |   6     |
| fare        |     891 |      nan | nan         |    nan |  32.2042   |  49.6934   |   0    |   7.9104 |  14.4542 |    31 | 512.329 |
| embarked    |     889 |        3 | S           |    644 | nan        | nan        | nan    | nan      | nan      |   nan | nan     |
| class       |     891 |        3 | Third       |    491 | nan        | nan        | nan    | nan      | nan      |   nan | nan     |
| who         |     891 |        3 | man         |    537 | nan        | nan        | nan    | nan      | nan      |   nan | nan     |
| adult_male  |     891 |        2 | True        |    537 | nan        | nan        | nan    | nan      | nan      |   nan | nan     |
| deck        |     203 |        7 | C           |     59 | nan        | nan        | nan    | nan      | nan      |   nan | nan     |
| embark_town |     889 |        3 | Southampton |    644 | nan        | nan        | nan    | nan      | nan      |   nan | nan     |
| alive       |     891 |        2 | no          |    549 | nan        | nan        | nan    | nan      | nan      |   nan | nan     |
| alone       |     891 |        2 | True        |    537 | nan        | nan        | nan    | nan      | nan      |   nan | nan     |

## Missing Values and Cleaning Decisions

|             |   missing_count |   missing_percent |
|:------------|----------------:|------------------:|
| deck        |             688 |             77.22 |
| age         |             177 |             19.87 |
| embarked    |               2 |              0.22 |
| embark_town |               2 |              0.22 |

- Dropped 2 rows for embarked, embark_town because each had under 5% missing values.
- Imputed age with median 28.0 because it had 19.87% missing values, which is in the 5%-30% rule range.
- Encoded deck missing values as 'Missing' because deck had 77.22% missing values, too high for reliable imputation.

## Univariate Analysis

| column   |   outlier_count |   lower_bound |   upper_bound |
|:---------|----------------:|--------------:|--------------:|
| age      |              65 |          2.5  |         54.5  |
| fare     |             114 |        -26.76 |         65.66 |

Fare mean = 32.10, median = 14.45, and mode = 8.05. Because the mean is greater than the median and the median is greater than or equal to the mode, fare is right-skewed; a smaller number of very expensive tickets pulls the average upward.

## Bivariate Analysis

### Survival by Sex

| group   |   survival_rate_percent |
|:--------|------------------------:|
| female  |                   74.04 |
| male    |                   18.89 |

### Survival by Passenger Class

| group    |   survival_rate_percent |
|:---------|------------------------:|
| pclass_1 |                   62.62 |
| pclass_2 |                   47.28 |
| pclass_3 |                   24.24 |

### Survival by Sex and Passenger Class

| sex    |   pclass |   survival_rate_percent |
|:-------|---------:|------------------------:|
| female |        1 |                   96.74 |
| female |        2 |                   92.11 |
| female |        3 |                   50    |
| male   |        1 |                   36.89 |
| male   |        2 |                   15.74 |
| male   |        3 |                   13.54 |

## Correlation Matrix

The heatmap uses exactly survived, pclass, age, sibsp, parch, and fare. The derived boolean columns adult_male and alone are excluded.

|          |   survived |   pclass |    age |   sibsp |   parch |   fare |
|:---------|-----------:|---------:|-------:|--------:|--------:|-------:|
| survived |      1     |   -0.336 | -0.07  |  -0.034 |   0.083 |  0.255 |
| pclass   |     -0.336 |    1     | -0.337 |   0.082 |   0.017 | -0.548 |
| age      |     -0.07  |   -0.337 |  1     |  -0.233 |  -0.171 |  0.094 |
| sibsp    |     -0.034 |    0.082 | -0.233 |   1     |   0.415 |  0.161 |
| parch    |      0.083 |    0.017 | -0.171 |   0.415 |   1     |  0.218 |
| fare     |      0.255 |   -0.548 |  0.094 |   0.161 |   0.218 |  1     |

### Two Strongest Off-Diagonal Correlations

| feature_pair   |   correlation |   absolute_correlation |
|:---------------|--------------:|-----------------------:|
| pclass vs fare |        -0.548 |                  0.548 |
| sibsp vs parch |         0.415 |                  0.415 |

The strongest pair is pclass vs fare with correlation -0.548. The second strongest pair is sibsp vs parch with correlation 0.415. These values summarize the two largest linear relationships among the selected numeric columns.

## Multivariate Data Story

![Survival by sex and class](charts/story_survival_by_sex_class.png)

Women show a much higher survival rate than men across passenger classes. First-class passengers also survive more often than lower-class passengers, suggesting that both gender and class shaped access to safety.

![Age by survival and sex](charts/story_age_by_survival_sex.png)

The age chart shows that survival patterns differ by sex across a wide age range. It also shows that age alone does not explain survival as strongly as sex and class do.

![Fare by class and survival](charts/story_fare_by_class_survival.png)

Higher fares are concentrated in first class, and first-class passengers have better survival outcomes. Fare is therefore partly a proxy for class and access to better locations or resources on the ship.

![Survival by family size](charts/story_family_size_survival.png)

Passengers traveling in small family groups tend to do better than passengers traveling alone or in larger groups. This suggests that moderate family support may have helped, while large groups may have been harder to coordinate during evacuation.

## Exploratory Standardization Check

| column   |   before_mean |   before_std |   after_mean |   after_std |
|:---------|--------------:|-------------:|-------------:|------------:|
| age      |       29.3152 |      12.9776 |            0 |           1 |
| fare     |       32.0967 |      49.6695 |            0 |           1 |

The standardized age and fare columns have means near 0 and standard deviations near 1. This check is only for EDA; the modeling pipeline performs its own train-only scaling later.
