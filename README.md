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

**Table 3:** _Regression Models and Train/Test Accuracies_

![image](https://github.com/WilliamChenn/Ozone_v._Melanoma_Analysis/assets/85557718/141f5ddc-0182-46b4-860e-509200bc33ff)

Upon evaluating the train accuracies for each model, we identified that Linear, Ridge,
and Elastic Net regression were all severely overfitted. Although the Least Absolute Shrinkage and Selection Operator (LASSO) regression was also overfitting, train accuracy and test accuracy of 0.47 and 0.28, respectively, proved to be the most accurate in the given circumstances. The Least Absolute Shrinkage and Selection Operator (LASSO) regression is employed for more prediction accuracy in which shrinkage pulls data points towards the mean (Ranstam J. & Cook J. A., 2018).

**Figure 3:** LASSO Regression Equation

![image](https://github.com/WilliamChenn/Ozone_v._Melanoma_Analysis/assets/85557718/68129580-e39a-4a2a-a247-232e41c267f9)
Source: Dwarki (2022, July 18)

Lambda (λ) is a tuning parameter that is selected prior to the Cross-validation technique.
LASSO uses |β| to penalize the high coefficients (Dwarki, 2022). We chose to proceed with our investigation using The Least Absolute Shrinkage and Selection Operator (LASSO) regression as its computational methods account for the overfitting challenges we encountered while applying other forms of regression.

# IV. Results

## Descriptive Analysis Results

**Figure 4:** Melanoma Cases/Rates and Ozone Concentration/Hole Area from 1979 - 1994

![image](https://github.com/WilliamChenn/Ozone_v._Melanoma_Analysis/assets/85557718/393da3a9-2e7a-43e6-a9dc-e70e537793ee)

_Note._ The line graphs above display the trends in melanoma cases/rates and the ozone concentrations/hole area from 1979 - 1994 in New Zealand.

**Figure 5:** Melanoma Cases/Rates and Ozone Concentration/Hole Area from 1996 - 2018

![image](https://github.com/WilliamChenn/Ozone_v._Melanoma_Analysis/assets/85557718/37eacb43-1857-43b6-9d8a-57d6aff60540)

_Note._ The line graphs above display the trends in melanoma cases/rates and the ozone concentrations/hole area from 1996 - 2018 in New Zealand.


As seen, there is an apparent increase in both the cases and rate of melanoma along with a decrease in ozone concentrations due to the ozone hole opening from 1979 - 1994. As such, these trends convey how ozone depletion tends to cause an increase in melanoma rates, as observed in these line graphs. However, from 1996 - 2018, there did not seem to be a trend similar to the one before, which led us to suspect that there were potential inaccuracies within the data. This is important because it confirms the results discussed in past research findings in the literature review section and raises further questions about possible external factors that may influence the continued trend of cancer cases.


**Figure 6:** Correlation of Variables from 1979 - 1994

![image](https://github.com/WilliamChenn/Ozone_v._Melanoma_Analysis/assets/85557718/9425d41e-9ed7-4076-87c9-471415a4356d)

_Note._ The scatter plots above display the relationship between the 4 different variables in the dataset from 1979 - 1994 in New Zealand.

**Figure 7:** Correlation of Variables from 1996 - 2018

![image](https://github.com/WilliamChenn/Ozone_v._Melanoma_Analysis/assets/85557718/03510635-a51d-43f8-afc1-44f20310231f)

_Note._ The scatter plots above display the relationship between the 4 different variables in the dataset from 1996 - 2018 in New Zealand.

As shown in Figure 6, the positive correlation between the cases of melanoma/rate of melanoma per 100k and the size of the ozone hole in 1979-1994 is relatively strong – with Pearson correlation coefficients of 0.76 for melanoma cases v. size of the ozone hole and 0.67 for rate per 100k v. size of the ozone hole. This indicates that the growing size of the Ozone hole is correlated with the increased amount of melanoma cases in New Zealand. Moreover, Figure 6 shows a strong negative correlation between the concentration of ozone in the stratosphere and the cases of melanoma/rate of melanoma per 100k in 1979-1994 –– with Pearson correlation coefficients of -0.77 for melanoma cases v. ozone concentration and -0.68 for rate per 100k v. ozone concentration. Furthermore, the p-values of all the correlation models from 1979-1995 is <0.05, which indicates that the correlation is statistically significant. However, for 1996 - 2018, there is little correlation between the variables as the Pearson correlation coefficients are significantly lower than the graphs showcased in Figure 6; additionally, the p-values are significantly more significant than those of the graphs displayed in Figure 6. This confirms our second research question as the weak correlation indicates that external factors have a more substantial influence on melanoma cases than the ozone concentration in the stratosphere. As a result, we conducted additional research (Reference - “The Uses of Sunbeds in New Zealand” section) regarding possible external factors that may cause the increase in melanoma cases in New Zealand despite the ozone hole's recovery.


## Predictive Analysis Results

The results of our predictive analysis were underwhelming as the train/test accuracies were low. Although we achieved the highest regression score using the Least Absolute Shrinkage and Selection Operator (LASSO) regression, the train and test accuracies were 0.47 and 0.28, respectively (Refer to table 1), which are relatively low scores. We suspect that the low accuracies were due to two possible factors: external factors affecting the rate/cases of melanoma and the size of our dataset. As a result, we concluded that our dataset is not viable for predictive analysis as it contains too few observations and attributes – 39 and 4, respectively. However, future studies can further explore the novel approach of utilizing predictive modeling in investigating the relationship between ozone concentration.

# Discussion & Conclusion
In the 1979 - 1994 dataset, we identified a strong negative correlation between the rates of melanoma and the ozone concentration. Predictably, we identified a strong positive correlation between the rates of melanoma and the ozone hole size. We deemed these findings coherent as ozone is proven to absorb a significant amount of UVB radiation, which is the leading factor causing melanoma (discussed in our literature review). However, in the 1996 - 2018 dataset, we did not see a similar trend as the Pearson correlation coefficients in our correlation analysis were relatively low with an insufficient p-value. This prompted the suspicion of external factors that may have influenced the rate of melanoma as well as the insufficiency of our dataset.
Regarding our predictive analysis, we achieved the highest regression score using LASSO regression. However, the LASSO regression scores that we received were still relatively low. Therefore, we concluded that our dataset is not viable for predictive analysis as it contains few observations.
Based on these results, we also looked into future implications of this research, including the application of radioisotopes in the medical field regarding the utilization of radiation therapy used to treat cancer. Thus, we will not only grasp a better understanding of the relationship between ozone concentration and melanoma but also come up with solutions to treat it. This occurs through the discussion of implementing the Solution Cathode Glow Discharge (SCGD) in our research. Both of these applications will be further discussed in the Future Implications section.

## The Uses of Sunbeds in New Zealand
Considering possible factors that may have provoked the persistent increase in melanoma rates in New Zealand, we identified the increased use of sunbeds among individuals in New Zealand. According to Køster et al. (2009), the artificial light from sunbeds emits UV radiation, which poses a high risk for melanoma development in younger adults aged 15 - 59. 35% of Danish individuals reported in 1994 that they had used sunbeds in the past two years, along with an increase to 50% in 2004 (Køster et al., 2009). However, efforts to reduce sunbed usage through public health campaigns have taken place recently (Olsen et al., 2019). Hence, these efforts have led to the imposition of sunbed regulations on January 4th, 2017. These regulations made it illegal for individuals under the age of 18 to use a sunbed in New Zealand, according to the Ministry of Health (2022, January 31).
Based on the increase in sunbed usage in New Zealand, we anticipate sunbed usage likely influenced the rising rates of melanoma. However, we could not obtain any datasets regarding the use of sunbeds in New Zealand from 1996 - 2018; yet, the information presented in the articles we reviewed allowed us to capture the essence of possible external factors that caused the rise of melanoma rates.

## Limitations of the Study
The inability to collect enough data regarding ozone in the stratosphere and skin cancer significantly limited the depth of study in our research. We could not access information regarding ozone data before 1979 as detailed assessments of the ozone layer were not conducted and/or documented by the National Aeronautics and Space Administration (NASA). The lack of abundant quantities of both observations and attributes resulted in weak predictive modeling and
22
correlation graphs. Moreover, the lack of tuning in our predictive models may have affected the accuracy of our regression scores.

## Future Implications
The future implications of this research are looking into more datasets to make our predictive model more accurate. Our LASSO regression method did not run well because of such a small dataset. By doing additional research, we hope to come across datasets that provide more information about the future state of ozone concentrations. This will ultimately allow for more robust predictive accuracy so that we can predict future rates of melanoma.
Furthermore, the future implications also involve exploring and applying radioisotopes in the medical field to treat cancerous cells. Moreover, radioisotopes are especially helpful in the medical field, mainly being used in radiation therapy as a way to slow down the growth or completely eradicate potentially dangerous skin cancer cells (Zhao et al., 2016; Zhang et al., 2010). Radioisotopes emit these forms of radiation in particles; hence, they can be used to treat cancerous skin cells (Zhang et al., 2010). For example, according to Zhang et al. (2010), some of these radioisotopes that are commonly implemented in the treatment of cancer are iodine - 131(131I), yttrium-90 (90Y), copper-67 (67Cu), rhenium-186 (186Re), lutetium-177 (177Lu), and copper-64 (64Cu).

**Figure 8:** Graphic on the Use of Radioisotopes in Treatment of Cancerous Cells

![image](https://github.com/WilliamChenn/Ozone_v._Melanoma_Analysis/assets/85557718/06893c13-9707-43cc-9585-59ff5081b4ae)

Source: Zhang et al. (2010)

The way these radioisotopes can be discovered is by the implementation of the Solution Cathode Glow Discharge (SCGD). The SCGD is a device that utilizes mass spectrometry analysis to detect molecular emissions from various solutions with the end goal of being able to detect radioisotopes by optimization (Doroski et al., 2013). Mass spectrometry analysis is where the mass-to-charge ratio is measured for one or more particles and applied to measuring isotope ratios such as carbon and nitrogen (Brand, 1996). Optimization has been performed in past research on the SCGD to maximize molecular emissions present in solutions. However, work is still being done so that the SCGD can detect these isotope ratios discussed. Once the SCGD can detect these forms of isotopes, they can be applied in future works regarding radiation therapy as a solution for treating melanoma.

**Figure 9:** Graphic on Solution Cathode Glow Discharge Functions
![image](https://github.com/WilliamChenn/Ozone_v._Melanoma_Analysis/assets/85557718/29e6fd71-f857-4167-9836-e38722bda11b)

Source: Doroski et al. (2013)

Evidently, these implications to our research from our results will be topics that will need to be further explored in the future.

## Concluding Remarks
In retrospect, we deemed that we have gained a more robust interpretation of both melanoma and the ozone layer. We found appropriate answers to our research questions during this project through data analysis with Pandas and predictive modeling using machine learning.
Although we did not obtain the highest regression score through our predictive analysis, we decided to look more in-depth into any possible outside factors that may have influenced our data. A significant finding from our research is that there are external factors to consider when looking at the cases of melanoma, as it does not always directly correlate with ozone concentration. Thus, we will need to take into account key behavior and habitual factors, stated in the Ethical Considerations section, that may influence melanoma so that we are providing the most accurate conclusions to the public.

# References
Armstrong, B. K., & Kricker, A. (1995). Skin cancer. Dermatologic Clinics, 13(3), 583–594. https://doi.org/10.1016/s0733-8635(18)30064-0
Barnes, P.W., Williamson, C.E., Lucas, R.M. et al. (2019). Ozone depletion, ultraviolet radiation, climate change and prospects for a sustainable future. Nature Sustainability 2, 569–579. https://doi.org/10.1038/s41893-019-0314-2
Brand, W. A. (1996). High precision isotope ratio monitoring techniques in mass spectrometry. Journal of Mass Spectrometry, 31(3), 225–235. https://doi.org/10.1002/(sici)1096-9888(199603)31:3
Decker, C. G., & Webb, M. R. (2016). Measurement of sample and plasma properties in solution-cathode glow discharge and effects of organic additives on these properties. Journal of Analytical Atomic Spectrometry, 31(1), 311–318. https://doi.org/10.1039/c5ja00243e
Diepgen, T. L., & Mahler, V. (2002). The epidemiology of skin cancer. British Journal of Dermatology, 146(s61), 1–6. https://doi.org/10.1046/j.1365-2133.146.s61.2.x
D'Orazio, J., Jarrett, S., Amaro-Ortiz, A., & Scott, T. (2013). UV radiation and the skin. International Journal of Molecular Sciences, 14(6), 12222–12248. https://doi.org/10.3390/ijms140612222
Doroski, T. A., King, A. M., Fritz, M. P., & Webb, M. R. (2013). Solution–cathode glow discharge – optical emission spectrometry of a new design and using a Compact Spectrograph. Journal of Analytical Atomic Spectrometry, 28(7), 1090–1095. https://doi.org/10.1039/c3ja50092f
Dupont, E., Gomez, J., & Bilodeau, D. (2013). Beyond UV radiation: A skin under challenge. International Journal of Cosmetic Science, 35(3), 224–232. https://doi.org/10.1111/ics.12036
Dwarki, V. (2022, July 18). Ridge Regression vs Lasso Regression. Naukri Learning. Retrieved July 26, 2022, from
https://www.naukri.com/learning/articles/ridge-regression-vs-lasso-regression/
Hussein, M. R. (2005). Ultraviolet radiation and skin cancer: Molecular mechanisms. Journal of Cutaneous Pathology, 32(3), 191–205. https://doi.org/10.1111/j.0303-6987.2005.00281.x
Jones, W. O., Harman, C. R., Ng, A. K. T., & Shaw, J. H. F. (1999). Incidence of malignant melanoma in Auckland, New Zealand: Highest rates in the world. World Journal of Surgery, 23(7), 732–735. https://doi.org/10.1007/pl00012378
Køster, B., Thorgaard, C., Clemmensen, I. H., & Philip, A. (2009). Sunbed use in the Danish population in 2007: A cross-sectional study. Preventive Medicine, 48(3), 288–290. https://doi.org/10.1016/j.ypmed.2008.12.012
Miller, A. J., & Mihm, M. C. (2006). Melanoma. New England Journal of Medicine, 355(1), 51–65. https://doi.org/10.1056/nejmra052166
MOH, (2021). Cancer: Historical summary 1948–2018. Ministry of Health NZ. . Retrieved July 23, 2022, from
https://www.health.govt.nz/publication/cancer-historical-summary-1948-2018
Newman, P. (2018). NASA Ozone Watch: Latest Status of Ozone. NASA. Retrieved July 23, 2022, from https://ozonewatch.gsfc.nasa.gov/statistics/annual_data.
Olsen, C. M., Green, A. C., Pandeya, N., & Whiteman, D. C. (2019). Trends in melanoma incidence rates in eight susceptible populations through 2015. Journal of Investigative Dermatology, 139(6), 1392–1395. https://doi.org/10.1016/j.jid.2018.12.006
Ranstam, J., & Cook, J. A. (2018). Lasso regression. British Journal of Surgery, 105(10), 1348–1348. https://doi.org/10.1002/bjs.10895
Saladi, R. N., & Persaud, A. N. (2005). The causes of Skin cancer: A comprehensive review. Drugs of Today, 41(1), 37. https://doi.org/10.1358/dot.2005.41.1.875777
Sneyd, M. J., & Cox, B. (2013). A comparison of trends in melanoma mortality in New Zealand and Australia: The two countries with the highest melanoma incidence and mortality in the world. BMC Cancer, 13(1). https://doi.org/10.1186/1471-2407-13-372
Solomon, S., Mills, M., Heidt, L. E., Pollock, W. H., & Tuck, A. F. (1992). On the evaluation of ozone depletion potentials. Journal of Geophysical Research, 97(D1), 825–842. https://doi.org/10.1029/91jd02613
Sunbeds. Ministry of Health NZ. (2022, January 31). Retrieved July 27, 2022, from https://www.health.govt.nz/your-health/healthy-living/environmental-health/sunbeds#:~:t ext=From%204%20January%202017%2C%20it,a%20current%20passport
Taalas, P., Damski, J., Kyrö, E., Ginzburg, M., & Talamoni, G. (1997). Effect of stratospheric ozone variations on UV radiation and on tropospheric ozone at high latitudes. Journal of Geophysical Research: Atmospheres, 102(D1), 1533–1539. https://doi.org/10.1029/96jd02310
United States Environmental Protection Agency. (2021, October 7). Basic Ozone Layer Science. EPA. Retrieved July 26, 2022, from https://www.epa.gov/ozone-layer-protection/frequently-asked-questions-about-ozone-lay er#:~:text=The%20ozone%20layer%20lies%20approximately,Earth's%20surface%2C%2 0in%20the%20stratosphere.
Urbach, F. (1980). Ultraviolet radiation and skin cancer in man. Preventive Medicine, 9(2), 227–230. https://doi.org/10.1016/0091-7435(80)90080-8
Zhang, L., Chen, H., Wang, L., Liu, T., Yeh, J., Lu, G., Wang, L., & Mao, H. (2010). Delivery of therapeutic radioisotopes using nanoparticle platforms: Potential Benefit in Systemic Radiation therapy. Nanotechnology, Science and Applications, (3), 159–170. https://doi.org/10.2147/nsa.s7462
Zhao, J., Zhou, M. & Li, C. Synthetic nanoparticles for delivery of radioisotopes and radiosensitizers in cancer therapy. Cancer Nano 7, 9 (2016). https://doi.org/10.1186/s12645-016-0022-9
