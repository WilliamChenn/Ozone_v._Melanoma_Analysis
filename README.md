# Ozone_v._Melanoma_Analysis
Project Presentation: https://docs.google.com/presentation/d/1NUrXpARbfYUEleM5kn4CVs9BD6nU360qjHn2lP3g0t0/edit?usp=sharing

# I. Abstract
Correlations have been identified in the past between ozone concentration and UV radiation. This led us to suspect a correlation between the ozone concentration and the increase of melanoma rates over the years in New Zealand, alongside other external factors. Our research explores the correlation between the atmospheric ozone concentration and the rates of melanoma in New Zealand. We also utilized predictive modeling (LASSO) in an attempt to predict rates of melanoma using ozone data. Findings from our research indicate a strong negative correlation between the ozone concentration and the rates of melanoma in New Zealand. Our research results also indicate that external factors also significantly influence the melanoma rates in New Zealand. Furthermore, we deemed our dataset unviable for predictive analysis as it contains few attributes and observations. Lastly, future applications of radioisotopes were considered to alleviate melanoma rate through the use of the Solution Cathode Glow Discharge.
Keywords: Melanoma, Ozone, New Zealand


# II. Introduction
The ozone layer is a protective region of the Earth’s stratosphere, lying roughly 15-40 kilometers above the surface of the Earth, according to the United States Environmental Protection Agency (2021). Fluctuations in the ozone concentration in the stratosphere largely influence the amount of ultraviolet (UV) radiation, specifically UVB rays, that make contact with the Earth’s surface (Hussein, 2005). This poses a severe threat to our society, as long-term UV exposure can lead to many health risks such as sunburn, premature aging, and, most importantly, melanoma. Melanoma is the most malignant form of skin cancer developed from melanocytes (cells that produce melanin), accounting for 80% of deaths from skin cancer (Miller & Mihm, 2006).

New Zealand melanoma cases and rates have increased dramatically since the 1960s (Jones et al., 1999). The underlying cause of the disproportionate rate of melanoma cases in New Zealand compared to the rest of the world is its geographic location. New Zealand is one of the countries closest to the Antarctic ozone hole; hence, New Zealand experiences high levels of ambient UV radiation, with UV indices ranging between 15 and 11 in big cities during the summer season (Sneyd & Cox, 2013). This dramatic proportion of skin cancer cases is important to consider as these skin cancer cases are a result of the depletion of our ozone layer from climate change, which allows for increased UV radiation to penetrate the Earth (Armstrong & Kricker, 1995; D’Orazio et al., 2013). Thus, the purpose of this study is to evaluate the relationship between ozone concentration in the stratosphere and annual melanoma rates in New Zealand and to predict future instances of melanoma using predictive Machine Learning models. Through this process, we hope to develop a comprehensive understanding of how ozone concentrations correlate to the cases of skin cancer alongside other factors to evaluate and forecast future rates of melanoma based on the fluctuations of ozone concentration in the stratosphere to research creative solutions for treating cancer. Furthermore, assessing the predictive models will provide necessary information for estimating the severity of melanoma rates in the future.

## Research Questions
This research project aims to address the adverse effects of climate change on our planet. Specifically, we are inquisitive concerning the possible trends between ozone concentration in New Zealand and melanoma rates. Furthermore, suppose a substantial negative correlation between ozone concentration and melanoma rates exists. In that case, our pursuit will be to develop a machine-learning model that accurately predicts the rates of melanoma given the ozone concentration. Thus, the research questions that lie at the foundation of our investigation and guide our study are as follows:
  1.  Does ozone concentration correlate with the melanoma rates in New Zealand?]
  2.  Do other factors (genetics, chemical exposure, tanning beds, etc.) have a stronger correlation to the cases of melanoma rather than ozone concentration?
As such, these research questions serve as the driving factor for our analysis which will
be further discussed later in the paper.

## Literature Review
Our research is concentrated on identifying the correlation between ozone concentration and the melanoma rates in New Zealand. In the analysis of the related literature, we identified three key themes in these related studies:
  1. The significance of ozone depletion
  2. The increase of UV radiation as a result of ozone depletion
  3. The association between melanoma and UV radiation
These related research studies provided both contexts to our investigation and a
foundation for our study; however, these studies lack the application of predictive modeling. Thus, our study will utilize machine learning algorithms to generate predictive models regarding the rate of ozone depletion with respect to the rate of melanoma, which will provide valuable insights into this topic of study

### The Significance of Ozone Depletion
Ozone depletion has been a significant climate concern in the past and the present (Barnes et al., 2019). The research findings from the three studies examine the significance of ozone depletion. For example, scientists in the early 1970s have shown a potential threat to the ozone layer from using chlorofluorocarbons (CFCs) alongside other chemical pollutants (Barnes et al., 2019). Additionally, the observations of these chemical pollutants, such as CFCs, provided by Solomon et al. (1992) have been used to develop models that display the relationship between the use of such chemical pollutants to trace the ozone depletion potentials. These models have been applied in studies of Antarctica to determine the amount of ozone that has been lost (Taalas et al., 1997).

### Increased UV Radiation as a Result of Ozone Depletion
The ozone layer is essential to supporting life on earth as it is a significant determinant of UV radiation's abundance and spectral distribution in the earth’s troposphere (Urbach, 1980). UV rays are divided into three categories: UVC, UVB, and UVA. High-energy UVC radiation (200-280 nm) is the most harmful of the three as it can kill unicellular organisms at the time of exposure; fortunately, the atmospheric ozone layer absorbs 99 percent of this radiation (Dupont et al., 2013). The ozone layer, however, only partly filters UVB rays (280-315 nm) – absorbing roughly 90 percent of the radiation. UVB rays have biological effects as they penetrate the epidermis, producing damaging reactive oxygen and nitrogen species (ROS and RNS), which cause inflammation, sunburn, and premature skin aging (Dupont et al., 2013). Lastly, the ozone only absorbs roughly 50 percent of UVA radiation (325–400 nm), which, although less energetic than UVB rays, are present in more considerable amounts and reach the dermis of the skin (Dupont et al., 2013). Evidently, ozone depletion in our atmosphere will increase the emission of UV radiation reaching the earth’s surface (Saladi & Persaud, 2005).

### Association between Skin Cancer and UV Radiation
Prolonged exposure to UV radiation can have detrimental effects on health, most notably, skin cancer (Armstrong & Kricker, 1995; D’Orazio et al., 2013). According to a study conducted by D’Orazio et al. (2013) and Armstrong & Kricker (1995), long-term exposure to UV radiation leads to the mutation of basal carcinoma, squamous carcinoma, and deadly melanoma. Interestingly, these cases of skin cancer hold a higher proportion in the white population (Armstrong & Kricker, 1995). Furthermore, in their analysis of the connection between skin cancer and UV radiation, Hussein (2005) discovered that UVB, one of the three forms of UV radiation, is most responsible for the mutation of genes in the human skin. Likewise, Diepgen and Mahler (2002) expanded upon previous studies regarding skin cancer in America and identified the same relationship between exposure to UVB radiation and skin cancer. Based on these findings, we can conclude that ozone depletion poses a direct and significant threat to our society. These studies serve as a foundation for our research as they provide contextual evidence for what our analysis is based upon. These research studies demonstrate the significance of our study and allow us to build upon this field of study.

# III. Materials and Methods
Datasets utilized for our research investigations were collected from the Goddard Space Flight Center of the National Aeronautics and Space Administration (NASA) and the New Zealand Cancer Registry (NZCR). The dataset, including information regarding the ozone, is collected by the Nasa Ozone Watch project at the Goddard Space Flight Center. NASA provided information that covers the annual mean ozone hole area, and the minimum of the Southern Hemisphere mean ozone composition, measured in million km2 and Dobson units (DU), respectively. The dataset covers these measurements from 1979 to 2021, with the exception of 1995, where data was not collected due to satellite instrument failure that year. Taking account of the missing ozone data in 1995, we decided to exclude information collected in 1995 from our other dataset to maintain consistency.
Regarding the dataset covering melanoma rates in New Zealand, the data is sourced from the NZCR and the New Zealand Mortality Collection. The data covers the annual count and rate of melanoma in New Zealand. Rates in this dataset are expressed per 100,000 individuals and are age-standardized rates (ASR) according to the World Health Organization (WHO) world standard population.

Data analysis and cleaning were done using the Pandas library. All data processing was performed in Jupyter Notebook, an open-sourced integrated development environment (IDE). Libraries that we utilized include Sklearn, Matplotlib, Pandas, SciPy, and Seaborn. Methods involved in our procedure were first cleaning the data by filtering out the unnecessary columns in the New Zealand Cancer Registry (NZCR) dataset with the drop method in Python. Next, we utilized Matplotlib in the descriptive analysis section of our research, in which we plotted line graphs and scatter plots to identify any possible trends in the data. Finally, we implemented machine learning through Lasso regression for the predictive analysis section, where we determined the predictability and accuracy of our statistical models. These methods will be covered more in-depth in the upcoming sections.

## Data Cleaning Methods
The Pandas package from Python was used in the data-cleaning process of our research. We dropped unnecessary columns for our analysis as the New Zealand Cancer Registry dataset included rates from various cancer types. Our data processing allowed us to efficiently analyze our dataset thereafter.

**Table 1:** Table of Variables with Values from 1979 - 1994

<img width="537" alt="image" src="https://github.com/WilliamChenn/Ozone_v._Melanoma_Analysis/assets/85557718/ca9059d4-45f1-41d2-b786-86cdace4eb89">

_Note._ The table above displays the data from 1979 - 1994 after cleaning the datasets with the Pandas drop method.

**Table 2:** Table of Variables with Values from 1996 - 2018

<img width="487" alt="image" src="https://github.com/WilliamChenn/Ozone_v._Melanoma_Analysis/assets/85557718/6e0c88b4-77e8-441b-a6cf-099a250d305a">

_Note._ The table above displays the data from 1996 - 2018 after cleaning the datasets with the Pandas drop method.

As seen, the tables that were dropped with the Pandas drop method removed all of the unnecessary columns in the original dataset for the New Zealand Cancer Registry and allowed us to focus solely on the rates of melanoma.

## Predictive Analysis Methods
Our process of identifying a particular predictive model is influenced by the datatypes of our target and predictive attributes. Since our dataset consists of numeric values and our research aims to predict quantifiable cases/rates of melanoma, we utilized regression models in our predictive analysis. We initially considered the following regression models for our research:
  ● Least Absolute Shrinkage and Selection Operator (LASSO)
  ● Linear Regression 
  ● Ridge
  ● ElasticNet
For the entirety of our investigation, we implemented a train/test split of 70:30, respectively since our dataset contains a relatively small quantity of observations. Furthermore, after the trial of 90:10 and 80:20 splits, which are also conventional train/test splits alongside 70:30, we achieved weaker model scores compared to 70:30 (See Figure 2).

**Figure 2:** _Train_test_split_

![image](https://github.com/WilliamChenn/Ozone_v._Melanoma_Analysis/assets/85557718/bcdcaa4b-5dfc-4ad0-8494-13f15b59cd88)



