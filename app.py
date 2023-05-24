import streamlit as st
from joblib import dump, load
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter


model_xgboost = load("models\optimal_xgb.joblib")


st.set_page_config(layout="wide")

st.sidebar.image("img\logo_accident.png", use_column_width=False)


def page_intro():
    st.markdown(
        "<h2 style='text-align: center; color: black;'>Road accident severity</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h3 style='text-align: center; color: black;'>How to reduce hospitalization and deaths caused by road accidents ?</h3>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h4 style='text-align: center; color: black;'>Context </h4>",
        unsafe_allow_html=True,
    )

    st.write(
        """
    3 actors from the French government with different motivations had a discussion on how to reduce severe accidents. \n
    The transport minister wants to reduce by all means, the number of severe accidents and is indifferent to the budget 
    allocated to related decisions. The economy minister wants to optimize expenses for reducing severe accidents.
    Finally, the analyst expert suggests an in-between solution where expenses would both reduce the number of severe accidents 
    without being related to too many minor accidents.
    """
    )
    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        st.write(" ")

    with col2:
        st.image("img\image_roles.png", use_column_width=True)

    with col3:
        st.write(" ")

    st.markdown(
        "<h4 style='text-align: center; color: black;'>Objective </h4>",
        unsafe_allow_html=True,
    )

    st.write(
        """
    The objective of this project was to first design a classification model with best overall performances to predict 
    severe accidents (at least one hospitalized or dead victim) and then to calibrate it for the purpose of all 3 actors 
    from the government. In addition to these objectives, we looked into details, which factors each actor would consider. 
    """
    )

    st.markdown(
        "<h4 style='text-align: center; color: black;'>Datas </h4>",
        unsafe_allow_html=True,
    )

    st.write(
        """
    We used the data from traffic regulation interdepartmental national observatory 
    (“Observatoire national interministériel de la sécurité routière”), self-service provided and extracted 
    from BAAC documents ([“Bulletins d’Analyse des Accidents Corporels de la circulation”](https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2021/)). 
    Four tables were documented between 2005 and 2020 with different types of information: 
    “caracteristics”, “places”, “users”, “vehicles”.
    """
    )

    col4, col5, col6 = st.columns([1, 3, 1])

    with col4:
        st.write(" ")

    with col5:
        st.image("img\descr_table.png", use_column_width=True)

    with col6:
        st.write(" ")


def page_viz():
    st.markdown(
        "<h2 style='text-align: center; color: black;'>Data Vizualisation </h2>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Target", "Geographic", "Visibility", "Vehicle", "Relation between variables"]
    )

    with tab1:
        st.markdown(
            "<h4 style='text-align: center; color: black;'>Target variable </h4>",
            unsafe_allow_html=True,
        )

        st.write(
            """
        We decided to define the target variable as the “presence of at least one hospitalized/dead victim during an accident”.
        """
        )

        col22, col23, col24 = st.columns([1, 4, 1])

        with col22:
            st.write(" ")

        with col23:
            st.image("img/targetVariable.png", use_column_width=True)

        with col24:
            st.write(" ")

        st.write(
            """
        Among all victims, we can observe that there are very few deaths (3%) but notably an important proportion of hospitalized victims (20%). 
        When we keep the most severe form of injury by accident, we end up with a balanced prevalence of 42% of accidents with at least one 
        hospitalized/dead victim. \n
        First, we assessed visually the link between the explanatory variables against the most severe injury by accident. 
        When we think about accidents, we can think of several causes impacting the severity, including for example: 
        geographic, visibility and vehicle information.
        """
        )

    with tab2:
        st.markdown(
            "<h4 style='text-align: center; color: black;'>Geographic information </h4>",
            unsafe_allow_html=True,
        )

        st.write(
            """
        Defining severity of accident from 1 to 4, 1 being no injury and 4 being death, we can have an overview of the mean severity of accident 
        by department. We also compared the mean severity of these accidents to the population density. 
        """
        )

        col19, col20, col21 = st.columns([1, 6, 1])

        with col19:
            st.write(" ")

        with col20:
            st.image("img/density_bis.png", use_column_width=True)

        with col21:
            st.write(" ")

        st.write(
            """
        We can observe that the densest department (Paris, Marseille, Lyon) have a low mean severity of injury. 
        Conversely, departments with the worst mean severity of injury (Mayenne, Landes) are the least dense. 
        It seems that the severity is negatively correlated to population density.
        \n \n
        The department density information is linked to the road category, for example there are more national roads 
        near small cities and more departmental roads in big cities.
        """
        )
        col25, col26, col27 = st.columns([1, 4, 1])

        with col25:
            st.write(" ")

        with col26:
            st.image("img/severityAgainstRoad.png", use_column_width=True)

        with col27:
            st.write(" ")

        st.write(
            """
        This intuition seems correct since the communal roads have far less cases of death than departmental or national ones. 
        The type of road is also linked to the type of agglomeration.
        """
        )

        col60, col61, col62 = st.columns([1, 3, 1])

        with col60:
            st.write(" ")

        with col61:
            st.image("img/agglo_catr.png", use_column_width=True)

        with col62:
            st.write(" ")

        col63, col64, col65 = st.columns([1, 2, 1])

        with col63:
            st.write(" ")

        with col64:
            st.image("img/agg_grav.png", use_column_width=True)

        with col65:
            st.write(" ")

        st.write(
            """
        First, we can observe that almost all agglomeration roads are communal ones while outside of agglomeration, 
        it is mainly departmental roads. 
        Then we can see that inside of agglomerations, the risk of severe injury is much lower than outside of agglomerations. 
        """
        )

    with tab3:
        st.markdown(
            "<h4 style='text-align: center; color: black;'>Visibility information </h4>",
            unsafe_allow_html=True,
        )

        st.write(
            """
        When driving, the main sense is sight. The hour of the day and brightness are directly impacting the sense of sight.
        \n \n
        In the following part, we have a first figure displaying proportion of injury severity by hour of the day. 
        The second figure is displaying fold severity, for each hour of the day, we divide the proportion of injury severity 
        by the overall proportion of this injury severity, this results into a fold. Fold means ‘how many times more risk of being part 
        of an injury group at a specific hour compared to overall proportion of this severity’. For example, a fold of 2 for death injury 
        at 2am means that there are 2 times more risk to die for accidents at 2am compared to overall proportion. 

        """
        )
        col28, col29, col30 = st.columns([1, 4, 1])

        with col28:
            st.write(" ")

        with col29:
            st.image("img/severityAgainstHour.png", use_column_width=True)

        with col30:
            st.write(" ")
        st.write(
            """
        We can observe a trend from 10pm to 6am, where proportion of death and hospitalized cases are above overall proportion, 
        with a peak reached at 3 to 5 am. This suggests that the light but also the journey cause may have an indirect impact on severity. 
        Taking kids to school, being stuck in traffic jam or going on holiday during the night tired are activities happening at specific 
        hours of the day and for which we can expect a different level of attention from the driver.
        \n \n
        On the second figure, we can observe 2 times more risks of death during an accident between 2 and 5am.
        \n \n
        We then look at brightness during accidents.

        """
        )
        col31, col32, col33 = st.columns([1, 4, 1])

        with col31:
            st.write(" ")

        with col32:
            st.image("img/severityAgainstLuminosity.png", use_column_width=True)

        with col33:
            st.write(" ")
        st.write(
            """
        We observe a heavily riskier type of accident during the night when there is no light with a fold of 3.

        """
        )

    with tab4:
        st.markdown(
            "<h4 style='text-align: center; color: black;'>Vehicle information </h4>",
            unsafe_allow_html=True,
        )

        st.write(
            """
        When we think about injury severity, we think of protections and thus of vehicle type. 
        Being inside of a heavy protected vehicle may feel safer than being on a motorbike. However, being impacted by a 
        heavy vehicle like a train may represent a high risk of death whatever the vehicle driving during the collision.
        """
        )
        col66, col67, col68 = st.columns([1, 2, 1])

        with col66:
            st.write(" ")

        with col67:
            st.image("img/vehicles_severity.png", use_column_width=True)

        with col68:
            st.write(" ")

        st.write(
            """
        As expected, even though only a few accidents with trains were reported, they represent the type of vehicle with the largest 
        risk of death or hospitalization. 
        ‘HPV’ category corresponds to 3.5 metric tons vehicles, we can make the same hypothesis as for the trains (weight and speed).
        """
        )

    with tab5:
        st.markdown(
            "<h4 style='text-align: center; color: black;'>Relation between variables </h4>",
            unsafe_allow_html=True,
        )

        st.write(
            """
        We last assessed the link between each pair of explanatory variables. The rationale was to only keep one variable 
        when two or more had a strong link in order to use specific models handling only variables with low-moderate relation, 
        for example penalized logistic models like lasso or elastic-net.
        \n \n
        The following figure corresponds to a heatmap cross-referring pairs of explanatory variables, 
        Cramer’s V value is computed for this pair. This value is measure of association between two categorical variables, 
        ranging from 0 to 1, 0 being a low association and 1 being a strong association.

        """
        )
        col7, col8, col9 = st.columns([1, 4, 1])

        with col7:
            st.write(" ")

        with col8:
            st.image("img/VCramerHeatmaps.png", use_column_width=True)

        with col9:
            st.write(" ")

        st.write(
            """
        As expected, we were right about the relation between hour of the day and brightness, but we can also observe a strong relation 
        between the variable ‘agglomeration’ and ‘brightness’ one. On the second heatmap, we observe a strong relation between the road slope 
        and curvature. Indeed, there are rarely corners in slopes. On the last heatmap, some variable sharing the same information are 
        highlighted, for example victim position and user category.

        """
        )


def page_model():
    st.markdown(
        "<h2 style='text-align: center; color: black;'> Modelization </h2>",
        unsafe_allow_html=True,
    )

    tab6, tab7 = st.tabs(["Pre-Processing", "Tuning"])

    with tab6:
        st.markdown(
            "<h4 style='text-align: center; color: black;'>Preprocessing </h4>",
            unsafe_allow_html=True,
        )

        st.write(
            """
            Our three objectives before starting modelling, was to 1- Strengthen our dataset with new and appropriate variables, 
            2- Having appropriate variables for any of the models used and 3- Being able to launch quickly model tuning.
            \n \n
            In order to strengthen our dataset, we created numerous variables for which we suspected a link with our target variable, 
            including for example accident occurring during a work-free day, age of the driver during the accident, 
            number of inhabitants of the city where the accident occurred, number of victims on the same side as the impact collision.
            \n \n
            For the purpose of having appropriate variable for modelling, we then used “one-hot encoding” to replace all categorical 
            variables by bivariate 0/1 variables. One-hot encoding purpose is to give the same strength to each modality. 
            Values of 1 correspond to the presence of the modality while 0 values correspond to the absence of the modality.

        """
        )

        col37, col38, col39 = st.columns([1, 1, 1])

        with col37:
            st.write(" ")

        with col38:
            st.image("img/OHEncoding.png", use_column_width=True)

        with col39:
            st.write(" ")

        st.write(
            """
            In addition to one-hot encoding, we removed variables with too low completion prevalence, 
            variables for which filling could easily be biased, liked road width. 
            When several variables were too much related (Cramer’s V value>0.4), we only kept the one with the strongest 
            relationship with our target variable. Then we removed any row with missing values. 
            The rationale for these two last choices was to be able to launch any machine learning model on our dataset.
            \n \n
            The last pre-processing step was to reduce the number of features from our dataset in order to be able to 
            launch quickly model tuning. For this purpose, we used XGBoost feature importance and ranked each variable 
            ‘weight’, ‘gain’, ‘cover’, ‘total gain’ and ‘total cover’. We removed variables located the most times among 
            the last ranks until we only had 70 features remaining.

        """
        )
        #        col10, col11, col12 = st.columns([1, 3, 1])#

        #        with col10:
        #            st.write(" ")

        #        with col11:
        #            st.image("weakest_features.png", use_column_width=True)

        #       with col12:
        #          st.write(" ")

        st.write(
            """
            After these three pre-processing steps, we ended-up with a completely filled dataset with 70 features for 836 553 accidents. 
            Target variable prevalence remained at 42%.
            \n \n
            This complete dataset was split into two set: a training dataset with 669 242 accidents (80%) for initiating models,
            and a test set of 167 311 accidents (20%) for evaluating models.

        """
        )

    with tab7:
        st.markdown(
            "<h4 style='text-align: center; color: black;'>Tuning </h4>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<h5 style='text-align: center; color: black;'>Performance Metrics</h5>",
            unsafe_allow_html=True,
        )

        col13, col14, col15 = st.columns([1, 1, 1])

        with col13:
            st.write(" ")

        with col14:
            st.image("img/perf_metrics.png", use_column_width=True)

        with col15:
            st.write(" ")

        st.write(
            """        
        The predictions of a model are compared to actual accident values, this allows to compute predictive performance of a model. 
        Some models output a probability for an accident of being severe, a threshold is needed to transform this probability into 
        a binary prediction.)
            """
        )

        col210, col211, col212 = st.columns([1, 1.5, 1])

        with col210:
            st.write(" ")

        with col211:
            st.image("img/crosstab.png", use_column_width=True)

        with col212:
            st.write(" ")

        st.write(
            """        
            From this confusion matrix, we can compute several performance indicators. 
            """
        )

        col40, col41, col42 = st.columns([1, 4, 1])

        with col40:
            st.write(" ")

        with col41:
            st.image("img/metrics.png", use_column_width=True)

        with col42:
            st.write(" ")

        st.write(
            """        
            A ROC curve can be traced from all positive and negative recalls for each model threshold. 
            From this ROC curve, we can compute the AUC (Area Under the Curve), it corresponds to our primary metric of interest. 
            AUC ranges from 0.5 (random predictions) to 1 (perfect predictions), we want to maximize AUC.
            For this project, a large AUC would mean that we predict well accident severity for numerous thresholds.       
            """
        )
        col43, col44, col45 = st.columns([1, 1, 1])

        with col43:
            st.write(" ")

        with col44:
            st.image("img/roc_curve.png", use_column_width=True)

        with col45:
            st.write(" ")

        st.markdown(
            "<h5 style='text-align: center; color: black;'>Cross-Validation</h5>",
            unsafe_allow_html=True,
        )

        st.write(
            """        
        For model tuning, the idea is to train a model through cross-validation with the training set and to test its performances 
        on the test set. Cross-validation corresponds to splitting the training set into k folds, we train a model on k-1 
        of these folds and we validate the model on the remaining fold. This process is repeated k times in order to validate 
        models on each fold by computing a metric, here we use the AUC. The mean of the k metrics is computed and a training metric
        is outputted. It is important to reach the best metrics value on the independent test set for which none of the data was used 
        during the training process.  \n
        Example of cross-validation with k=5 folds and AUC metric. """
        )

        col200, col201, col202 = st.columns([1, 3, 1])

        with col200:
            st.write(" ")

        with col201:
            st.image("img/crossValidationExample.png", use_column_width=True)

        with col202:
            st.write(" ")

        st.write(
            """
        For time saving constraints, we decided to use 3 folds during cross-validation. 
        A larger number of folds could refine metrics estimation but at the cost of an extended code execution.

        We decided to maximize AUC performances on test set and then to pick the models with best AUCs, in order to finally 
        calibrate them for our case-study.
        
        """
        )

        st.markdown(
            "<h5 style='text-align: center; color: black;'>Model Tuning</h5>",
            unsafe_allow_html=True,
        )

        st.write(
            """        
        We decided to assess several classification machine learning methods for which algorithmic mechanism differ. 
        Here is a non-exhaustive list of advantages and drawback for each method.       
        """
        )

        col46, col47, col48 = st.columns([1, 4, 1])

        with col46:
            st.write(" ")

        with col47:
            st.image("img/algorithms.png", use_column_width=True)

        with col48:
            st.write(" ")

        st.write(
            """        
        We assessed methods with different hyperparameter values. Due to its very time-consuming aspect, the SVM has not been 
        retained for tuning."""
        )

        col49, col50, col51 = st.columns([1, 4, 1])

        with col49:
            st.write(" ")

        with col50:
            st.image("img/hyperparameters.png", use_column_width=True)

        with col51:
            st.write(" ")

        st.write(
            """ 
        Concerning the MLP, there is no parameter to tune, the model was initialized with two Dense layers, 35 neurones, 
        2 final classes and tanh/softmax activation functions.
        \n \n
        We can observe the best AUC values reached by each model after tuning.

        """
        )

        col52, col53, col54 = st.columns([1, 2, 1])

        with col52:
            st.write(" ")

        with col53:
            st.image("img/AUCComparison.png", use_column_width=True)

        with col54:
            st.write(" ")

        st.write(
            """        
        Best performances were observed for XGBoost, LGBM, MLP and Elastic Net after tuning.
        \n \n
        For ease of use, we decided to exclusively keep performing and low time-consuming for model comparison in the case study: 
        XGBoost, LGBM and Elastic Net.
        """
        )


def page_case_study():
    st.markdown(
        "<h2 style='text-align: center; color: black;'> Case Study </h2>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h5 style='text-align: center; color: black;'>Case-study problematic </h5>",
        unsafe_allow_html=True,
    )

    st.write(
        """
        In order to make a connection between our models and the case-study, we displayed the distribution of predictive 
        probabilities of XGBoost. These predictive probabilities are used for to compute confusion tables at each threshold of interest.
    """
    )
    col16, col17, col18 = st.columns([1, 4, 1])

    with col16:
        st.write(" ")

    with col17:
        st.image("img/XGBoost.png", use_column_width=True)

    with col18:
        st.write(" ")

    st.write(
        """
        As a reminder, transport minister wants to reduce by all means, the number of severe accidents, in other terms maximizing positive recall. Economy minister wants to optimize expenses for reducing severe accidents, meaning that we must maximize positive precision for this specific case. Finally, the analyst expert suggests an in-between solution where expenses would both reduce the number of severe accidents without being related to too many minor accidents. From a statistical point of view, a solution for the latest would be using a Youden threshold associated with a maximization of the sum of positive and negative recalls.
        \n \n
        In order to keep a reasonable decision, we set to 80%, the positive recall to reach for the transport minister and to 80% the positive precision to reach for the economy minister.
        \n \n
        Two questions arise:
        \n \n
        What thresholds and which associated performances would we reach for each government actor?
        \n \n
        Which model would outperform other ones and optimize each problematic performance?
    """
    )

    st.markdown(
        "<h5 style='text-align: center; color: black;'>Calibration </h5>",
        unsafe_allow_html=True,
    )

    st.write(
        """
        As an illustration of how to calibrate a model for each actor, we used the XGBoost predictions. 
        We displayed each performance criteria against XGBoost thresholds in order to select optimal threshold, as follows.    
        """
    )
    col85, col86, col87 = st.columns([1, 4, 1])

    with col85:
        st.write(" ")

    with col86:
        st.image("img/performanceCriteria.png", use_column_width=True)

    with col87:
        st.write(" ")

    st.markdown(
        "<h5 style='text-align: center; color: black;'>Performances </h5>",
        unsafe_allow_html=True,
    )

    st.write(
        """
        As an illustration of how to calibrate a model for each actor, we used the XGBoost predictions. 
        We displayed each performance criteria against XGBoost thresholds in order to select optimal threshold, as follows.    
        """
    )
    col55, col56, col57 = st.columns([1, 4, 1])

    with col55:
        st.write(" ")

    with col56:
        st.image("img/calibrated_performances.png", use_column_width=True)

    with col57:
        st.write(" ")

    st.write(
        """
    For any problematic, XGBoost and LGBM attained similar performances and were outperforming Elast-Net.
    \n \n
    Now, we can for example focus on XGBoost and look at predictions by feature for the economy minister as a representative example.
        """
    )

    st.markdown(
        "<h5 style='text-align: center; color: black;'>Decisions </h5>",
        unsafe_allow_html=True,
    )

    st.write(
        """
    Once the model is set, we can have a look at which features had good performances so that, actors could make an infrastructure 
    improvement proposition or change traffic regulation to reduce the number of severe accidents to the government. 
    We decided to look specifically at the economy minister case.
    """
    )
    col77, col78, col79 = st.columns([1, 1.5, 1])

    with col77:
        st.write(" ")

    with col78:
        st.image("img/best_precision.png", use_column_width=True)

    with col79:
        st.write(" ")

    st.write(
        """
             The 3 features with the best positive precision were ‘level crossing’ (88.6%), ‘3 or more vehicles involved in the accident’ 
    (84.4%) and ‘accident taking place outside of agglomeration’ (84.3%). 
    For some features, it is quite hard to take decisions concerning how to lower the proportion of severe accidents. 
    However, the economy minister could have suggested, for example, to assess the reason why level crossing and out-of-agglomeration 
    cases have inflated severe accidents. Several reasons could explain this, like insisting on reducing the speed next to a level 
    crossing, or reducing the speed in specific areas. 
        """
    )


def page_demo():
    st.markdown(
        "<h2 style='text-align: center; color: black;'>Demonstration </h2>",
        unsafe_allow_html=True,
    )

    st.write(
        """ 
            In this part, we let you choose the parameters of an accident.
            \n \n
            In return, you'll see the probability evaluated by our XGBoost model for this accident to be severe. 
            You'll also see which one of the three actors in our case study would consider this accident interesting for 
            any political measure.  
            \n \n 
            For a better readability, we've only showed the five parameters with the most influence on the result 
            but you can choose every parameter by expanding below.   
            
            
            """
    )

    st.markdown(
        "<h4 style='text-align: center; color: black;'>Model parameters </h4>",
        unsafe_allow_html=True,
    )

    # Conteneur des variables non visibles
    col58, col59, col60 = st.columns([1, 1, 1])

    with col58:
        population = st.selectbox(
            "City Size",
            [
                "village",
                "small town",
                "average city",
                "big city",
                "metropolis",
                "other",
            ],
        )
        categorie_route = st.selectbox(
            "Road Type",
            [
                "other",
                "national",
                "departmental",
                "communal",
            ],
        )

    with col59:
        conditions_atmospheriques = st.selectbox(
            "Weather conditions",
            [
                "light rain",
                "heavy rain",
                "smog",
                "dazzling",
                "cloudy",
                "other",
            ],
        )
        luminosité = st.selectbox(
            "brightness",
            [
                "dawn ou dusk",
                "night w/o public lighting",
                "night with public lighting",
                "other",
            ],
        )
    with col60:
        nb_vehicule = st.slider("number of vehicles", 0, 10, 2)

    with st.expander("other parameters", expanded=False):
        col61, col62, col63 = st.columns([1, 1, 1])

        with col61:
            age_moyen_conducteurs = st.slider("average age of drivers", 18, 100, 30)
            nb_personnes_cote_choc = st.slider(
                "number of person on the impact side", 0, 10, 1
            )

            regime_circulation = st.selectbox(
                "traffic type",
                [
                    "other",
                    "bidirectional",
                    "with separate ways",
                    "with specials ways",
                ],
            )
            type_collision = st.selectbox(
                "collision type",
                [
                    "other",
                    "3 vehicles et + - multiples",
                    "2 vehicles - back impact",
                    "2 vehicles - side impact",
                    "3 vehicles et + - chain impact ",
                    "w/o collision",
                ],
            )
            infra = st.selectbox(
                "road infrastructure",
                [
                    "other",
                    "interchange ramps",
                    "traffic circle",
                    "toll area",
                ],
            )

        with col62:
            intersection = st.selectbox(
                "intersection type",
                [
                    "other",
                    "railroad crossing",
                    "traffic roundabout ",
                    "crossroads",
                    "outside intersections",
                ],
            )

            mois = st.selectbox(
                "month",
                [
                    "march",
                    "january",
                    "february",
                    "july",
                    "august",
                    "october",
                    "december",
                    "other",
                ],
            )

            profil = st.selectbox("road profile", ["slope", "hill top", "other"])
            situation = st.selectbox(
                "accident situation",
                [
                    "other",
                    "on emergency lane",
                    "on shoulder",
                    "on sidewalk",
                    "on special lane",
                ],
            )
            surface = st.selectbox(
                "road surface",
                ["other", "with oil / greasy substance ", "wet"],
            )
        with col63:
            pres_2roues = st.checkbox("presence of a two wheeler")
            pres_EPD = st.checkbox("presence of a personal transporter")
            pres_PL = st.checkbox("presence of a HGV")
            pres_train = st.checkbox("presence of a train")
            pres_pieton = st.checkbox("presence of a pedestrian")
            abs_obstacle = st.checkbox("no obstacles")
            loc_pieton = st.checkbox(
                "pedestrian on pedestrian crossing w/o light signaling"
            )
            nuit = st.checkbox("night accident")
            route_rectiligne = st.checkbox("straight road")
            pres_homme_volant = st.checkbox("presence of a male driver")
            pres_femme_volant = st.checkbox("presence of a female driver")
            trajet_promenade = st.checkbox("promenade")
            pres_piste_cyclabe = st.checkbox("presence of a bike path")
        # ... Fin du conteneur

    result = st.button("Predict")
    if result:
        X_test = pd.DataFrame(
            columns=[
                "choc_cote",
                "ageMeanConductors",
                "nbVeh",
                "prof_2.0",
                "prof_3.0",
                "planGrp_1.0",
                "surf_2.0",
                "surf_8.0",
                "atm_2.0",
                "atm_3.0",
                "atm_5.0",
                "atm_7.0",
                "atm_8.0",
                "vospGrp_1.0",
                "catv_EPD_exist_1",
                "catv_PL_exist_1",
                "trajet_coursesPromenade_conductor_1",
                "sexe_male_conductor_1",
                "sexe_female_conductor_1",
                "intGrp_Croisement circulaire",
                "intGrp_Croisement de deux routes",
                "intGrp_Hors intersection",
                "intGrp_Passage à niveau",
                "catv_train_exist_1",
                "infra_3.0",
                "infra_5.0",
                "infra_7.0",
                "infra_9.0",
                "catr_2.0",
                "catr_3.0",
                "catr_4.0",
                "catr_9.0",
                "hourGrp_nuit",
                "lum_2.0",
                "lum_3.0",
                "lum_5.0",
                "circ_2.0",
                "circ_3.0",
                "circ_4.0",
                "nbvGrp_1",
                "nbvGrp_2",
                "nbvGrp_3",
                "nbvGrp_4+",
                "catv_2_roues_exist_1",
                "col_2.0",
                "col_3.0",
                "col_4.0",
                "col_5.0",
                "col_6.0",
                "col_7.0",
                "obsGrp_Pas d'Obstacle",
                "situ_2.0",
                "situ_3.0",
                "situ_4.0",
                "situ_6.0",
                "situ_8.0",
                "populationGrp_Grande Ville",
                "populationGrp_Métropole",
                "populationGrp_Petite Ville",
                "populationGrp_Village",
                "populationGrp_Ville Moyenne",
                "mois_label_aug",
                "mois_label_dec",
                "mois_label_fev",
                "mois_label_jan",
                "mois_label_jul",
                "mois_label_mar",
                "mois_label_oct",
                "etatpGrp_pieton_alone_1",
                "locpGrp_pieton_3_1",
            ],
            dtype="int",
        )

        X_test.loc[0, "choc_cote"] = nb_personnes_cote_choc
        X_test.loc[0, "ageMeanConductors"] = age_moyen_conducteurs
        X_test.loc[0, "nbVeh"] = nb_vehicule

        X_test.loc[0, "prof_2.0"] = 1 if profil == "slope" else 0
        X_test.loc[0, "prof_3.0"] = 1 if profil == "hill top" else 0

        X_test.loc[0, "planGrp_1.0"] = 1 if route_rectiligne else 0

        X_test.loc[0, "surf_2.0"] = 1 if surface == "wet" else 0
        X_test.loc[0, "surf_8.0"] = 1 if surface == "with oil / greasy substance" else 0

        X_test.loc[0, "atm_2.0"] = 1 if conditions_atmospheriques == "light rain" else 0
        X_test.loc[0, "atm_3.0"] = 1 if conditions_atmospheriques == "heavy rain" else 0
        X_test.loc[0, "atm_5.0"] = 1 if conditions_atmospheriques == "smog" else 0
        X_test.loc[0, "atm_7.0"] = 1 if conditions_atmospheriques == "dazzling" else 0
        X_test.loc[0, "atm_8.0"] = 1 if conditions_atmospheriques == "cloudy" else 0

        X_test.loc[0, "vospGrp_1.0"] = 1 if pres_piste_cyclabe else 0

        X_test.loc[0, "catv_EPD_exist_1"] = 1 if pres_EPD else 0
        X_test.loc[0, "catv_PL_exist_1"] = 1 if pres_PL else 0

        X_test.loc[0, "trajet_coursesPromenade_conductor_1"] = (
            1 if trajet_promenade else 0
        )

        X_test.loc[0, "sexe_male_conductor_1"] = 1 if pres_homme_volant else 0
        X_test.loc[0, "sexe_female_conductor_1"] = 1 if pres_femme_volant else 0

        X_test.loc[0, "intGrp_Croisement circulaire"] = (
            1 if intersection == "traffic roundabout" else 0
        )
        X_test.loc[0, "intGrp_Croisement de deux routes"] = (
            1 if intersection == "crossroads" else 0
        )
        X_test.loc[0, "intGrp_Hors intersection"] = (
            1 if intersection == "outside intersections" else 0
        )
        X_test.loc[0, "intGrp_Passage à niveau"] = (
            1 if intersection == "railroad crossing" else 0
        )

        X_test.loc[0, "catv_train_exist_1"] = 1 if pres_train else 0

        X_test.loc[0, "infra_3.0"] = 1 if infra == "interchange ramps" else 0
        X_test.loc[0, "infra_5.0"] = 1 if infra == "traffic circle" else 0
        X_test.loc[0, "infra_7.0"] = 1 if infra == "toll area" else 0
        X_test.loc[0, "infra_9.0"] = 1 if infra == "other" else 0

        X_test.loc[0, "catr_2.0"] = 1 if categorie_route == "national" else 0
        X_test.loc[0, "catr_3.0"] = 1 if categorie_route == "departmental" else 0
        X_test.loc[0, "catr_4.0"] = 1 if categorie_route == "communal" else 0
        X_test.loc[0, "catr_9.0"] = 1 if categorie_route == "other" else 0

        X_test.loc[0, "hourGrp_nuit"] = 1 if nuit else 0

        X_test.loc[0, "lum_2.0"] = 1 if luminosité == "dawn ou dusk" else 0
        X_test.loc[0, "lum_3.0"] = 1 if luminosité == "night w/o public lighting" else 0
        X_test.loc[0, "lum_5.0"] = (
            1 if luminosité == "night with public lighting" else 0
        )

        X_test.loc[0, "circ_2.0"] = 1 if regime_circulation == "bidirectional" else 0
        X_test.loc[0, "circ_3.0"] = (
            1 if regime_circulation == "with separate ways" else 0
        )
        X_test.loc[0, "circ_4.0"] = (
            1 if regime_circulation == "with specials ways" else 0
        )

        X_test.loc[0, "nbvGrp_1"] = 1 if nb_vehicule == 1 else 0
        X_test.loc[0, "nbvGrp_2"] = 1 if nb_vehicule == 2 else 0
        X_test.loc[0, "nbvGrp_3"] = 1 if nb_vehicule == 3 else 0
        X_test.loc[0, "nbvGrp_4+"] = 1 if nb_vehicule >= 4 else 0

        X_test.loc[0, "catv_2_roues_exist_1"] = 1 if pres_2roues else 0

        X_test.loc[0, "col_2.0"] = (
            1 if type_collision == "2 vehicles - back impact" else 0
        )
        X_test.loc[0, "col_3.0"] = (
            1 if type_collision == "2 vehicles - side impact" else 0
        )
        X_test.loc[0, "col_4.0"] = (
            1 if type_collision == "3 vehicles et + - chain impact " else 0
        )
        X_test.loc[0, "col_5.0"] = (
            1 if type_collision == "3 vehicles et + - multiples" else 0
        )
        X_test.loc[0, "col_6.0"] = 1 if type_collision == "w/o collision" else 0
        X_test.loc[0, "col_7.0"] = 1 if type_collision == "other" else 0

        X_test.loc[0, "obsGrp_Pas d'Obstacle"] = 1 if abs_obstacle else 0

        X_test.loc[0, "situ_2.0"] = 1 if situation == "on emergency lane" else 0
        X_test.loc[0, "situ_3.0"] = 1 if situation == "on shoulder" else 0
        X_test.loc[0, "situ_4.0"] = 1 if situation == "on sidewalk" else 0
        X_test.loc[0, "situ_6.0"] = 1 if situation == "on special lane" else 0
        X_test.loc[0, "situ_8.0"] = 1 if situation == "other" else 0

        X_test.loc[0, "populationGrp_Grande Ville"] = (
            1 if population == "big city" else 0
        )
        X_test.loc[0, "populationGrp_Métropole"] = (
            1 if population == "metropolis" else 0
        )
        X_test.loc[0, "populationGrp_Petite Ville"] = (
            1 if population == "small town" else 0
        )
        X_test.loc[0, "populationGrp_Village"] = 1 if population == "village" else 0
        X_test.loc[0, "populationGrp_Ville Moyenne"] = (
            1 if population == "average city" else 0
        )

        X_test.loc[0, "mois_label_aug"] = 1 if mois == "august" else 0
        X_test.loc[0, "mois_label_dec"] = 1 if mois == "december" else 0
        X_test.loc[0, "mois_label_fev"] = 1 if mois == "february" else 0
        X_test.loc[0, "mois_label_jan"] = 1 if mois == "january" else 0
        X_test.loc[0, "mois_label_jul"] = 1 if mois == "july" else 0
        X_test.loc[0, "mois_label_mar"] = 1 if mois == "march" else 0
        X_test.loc[0, "mois_label_oct"] = 1 if mois == "october" else 0

        X_test.loc[0, "etatpGrp_pieton_alone_1"] = 1 if pres_pieton else 0

        X_test.loc[0, "locpGrp_pieton_3_1"] = 1 if loc_pieton else 0

        proba = model_xgboost.predict_proba(X_test)

        limits = [32, 43, 63, 100]
        data_to_plot = (
            "Probability for the accident to be severe",
            round(proba[0, 1] * 100, 0),
        )
        palette = sns.color_palette("Blues", len(limits))
        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        ax.set_yticks([1])
        ax.set_yticklabels([data_to_plot[0]])

        prev_limit = 0
        for idx, lim in enumerate(limits):
            ax.barh(
                [1],
                lim - prev_limit,
                left=prev_limit,
                height=15,
                color=palette[idx],
            )
            prev_limit = lim

        ax.barh([1], data_to_plot[1], color="black", height=5)
        plt.annotate(
            "Transport \n Minister",
            xy=(32, 9),
            xytext=(23, 14),
            arrowprops={
                "facecolor": "blue",
                "linewidth": 0.5,
                "headwidth": 10,
                "headlength": 4,
            },
            size=6,
        )
        plt.annotate(
            "Data \n Expert",
            xy=(43, 9),
            xytext=(40, 14),
            arrowprops={
                "facecolor": "blue",
                "linewidth": 0.5,
                "headwidth": 10,
                "headlength": 4,
            },
            size=6,
        )
        plt.annotate(
            "Economy \n Minister",
            xy=(63, 9),
            xytext=(58, 14),
            arrowprops={
                "facecolor": "blue",
                "linewidth": 0.5,
                "headwidth": 10,
                "headlength": 4,
            },
            size=6,
        )
        plt.tight_layout()
        col95, col96, col97 = st.columns([1, 4, 1])

        with col95:
            st.write(" ")

        with col96:
            st.pyplot(fig=fig)

        with col97:
            st.write(" ")


def page_conclu():
    st.markdown(
        "<h2 style='text-align: center; color: black;'> Conclusion </h2>",
        unsafe_allow_html=True,
    )

    tab12, tab10, tab11 = st.tabs(["Conclusion", "Limits", "Areas of Improvements"])

    with tab12:
        col193, col194, col195 = st.columns([1, 4, 1])

        with col193:
            st.write(" ")

        with col194:
            st.image("img/conclusion.png", use_column_width=True)

        with col195:
            st.write(" ")

    with tab10:
        st.markdown(
            "<h5 style='text-align: center; color: black;'>Limits </h5>",
            unsafe_allow_html=True,
        )

        st.write(
            """
        Our dataset degree of information is limited. For instance, we can’t know which driver is responsible of the accident, 
        also some of the dataset variables’ completion may be imprecise (e.g. road width or presence of lights near the accident)
        \n \n
        Unlike recall, positive precision is not a monotone performance criteria. Positive precision is linked to positive observations 
        prevalence. If we compared our performances when calibrating on precision with another independent dataset, we would not be sure 
        to expect the same results.
        \n \n
        Let’s keep in mind that our model doesn’t predict traffic accidents but rather accident severity. This means that we could estimate 
        the degree of severity of an accident considering involved features when the accident unfortunately already occurred but we would 
        not be sure that the case-study actors’ decisions would have a direct impact on the proportion of severe accident.

        """
        )

    with tab11:
        st.markdown(
            "<h5 style='text-align: center; color: black;'>Areas of improvement</h5>",
            unsafe_allow_html=True,
        )
        col103, col104, col105 = st.columns([5, 1, 5])

        with col103:
            st.write(" ")

        with col104:
            st.image("img/multiple_imputation.png", use_column_width=True)

        with col105:
            st.write(" ")

        st.write(
            """

            During pre-processing, the completion was an important criteria for keeping a variable. Using a strong imputation method like 
            multiple imputation would have saved some variables and dataset rows. This could have had a positive impact on test predictions. 
            Comparing our current results to an imputation method strategy would be a nice way to verify this clue.
            """
        )
        col106, col107, col108 = st.columns([2, 1, 2])

        with col106:
            st.write(" ")

        with col107:
            st.image("img/feature_selection.png", use_column_width=True)

        with col108:
            st.write(" ")

        st.write(
            """
            During data pre-processing, we could have improved our feature selection decision to keep only 70 features. We decided to keep the 
            features most frequently ranked among last ones by XGBoost feature importance. One possible amelioration would have been to keep 
            the mean rank of each feature when we pool XGBoost feature importance.
            """
        )

        col100, col101, col102 = st.columns([1, 2, 1])

        with col100:
            st.write(" ")

        with col101:
            st.image("img/noise_features.png", use_column_width=True)

        with col102:
            st.write(" ")

        st.write(
            """
            We observed a strong impact of some features like the ‘accident occuring in a village’. This could be a problem. 
            Wrong predictions being caused by one strong feature could hide the other features expression and create a systematic bias. 
            An idea would have been to add noise to the ‘accident occuring in a village’ feature upstream, before modelling step. 
            This would have given less expression to this feature, and more expression to other ones (more variability and less bias).

        """
        )


page = st.sidebar.radio(
    "",
    [
        "Introduction",
        "Data Vizualisation",
        "Modelization",
        "Case Study",
        "Let's try our model !",
        "Conclusion",
    ],
)


if page == "Introduction":
    page_intro()
elif page == "Case Study":
    page_case_study()
elif page == "Let's try our model !":
    page_demo()
elif page == "Data Vizualisation":
    page_viz()
elif page == "Modelization":
    page_model()
elif page == "Conclusion":
    page_conclu()

st.divider()

st.sidebar.info(
    """Authors : [Yacine Hajji](https://www.linkedin.com/in/yacine-hajji-3b7b29113/), [Benjamin Papin](https://www.linkedin.com/in/benjamin-papin-66406ba0/) 
    Cohort Data Scientist June 2022, DataScientest"""
)

st.sidebar.image("img/logo_ds.png", use_column_width=False)
