
# **MexEE 402: Data Preprocessing Case Study**

*MexEE Elective 2: Data Science and Machine Learning*  
**Batangas State University, Alangilan Campus**  
*1st Semester, AY 2026–2027*

---

## **Members**
| **Name** | **Student Number** | **Section** |
|---|---|---|
| GARCIA, JHADE KIMBERLY A.| 23-03063 | MEXE - 4101 |
| PAGAR, CLAIRE ALLEN A. | 23-06741 | MEXE - 4101 |


---

## **Notebook Links**

| **Chapter** | **Assigned Member** | **Notebook Link** |
|---|---|---|
| **Ch1_2_3** | Garcia J. | [Open Notebook](https://colab.research.google.com/drive/1_D0gw-jE0bEKVSbDLPXuS23Y7HvKj1-7?usp=sharing) |
| **Ch4** | Garcia J. | [Open Notebook](https://colab.research.google.com/drive/1NkH3Pzi7dnScA3MzXwajQK2EfEC3oVVZ?usp=sharing) |
| **Ch5** | Garcia J. | [Open Notebook](https://colab.research.google.com/drive/1affjaCfe6tLF4BUhUPYHlt-nA852DIJV?usp=sharing) |
| **Ch6** | Pagar | [Open Notebook](https://colab.research.google.com/drive/1FUhxr4QARN_1R0Wqj2JhlhrxCpUKh1Lo?usp=sharing) |
| **Ch7** | Pagar | [Open Notebook](https://colab.research.google.com/drive/1EEU6NN0fbEGgdLhn010h1M2SFB6rXwi4?usp=sharing) |
| **Ch8** | Pagar | [Open Notebook](PASTE_CLAIRE_CH8_LINK_HERE) |
| **Ch9** | Pagar | [Open Notebook](https://colab.research.google.com/drive/1JH20JaN3H_SHar_KOKujbYp4kJicZddH?usp=sharing) |

---

## **What We Learned**

*Write one short paragraph per chapter, from Ch1_2_3 to Ch9. Explain what the chapter taught you, what you understood, and what surprised you. Focus on your own learning and understanding, not just on what the library does.*

### **Chapter 1–2–3: [Chapter Title]**

*Write your reflection here.*

### **Chapter 4: [Chapter Title]**

*Write your reflection here.*

### **Chapter 5: [Chapter Title]**

*Write your reflection here.*

### **Chapter 6: [Chapter Title]**

*Write your reflection here.*

### **Chapter 7: [Chapter Title]**

*Write your reflection here.*

### **Chapter 8: Constructing a Preprocessing Pipeline**

*Write your reflection here.*

### **Chapter 9: [Chapter Title]**

*Write your reflection here.*

---

## **Errors We Found**

*Each chapter was checked for errors in the original notebook. Any identified error is documented below, together with its correction and explanation. If no error was found, this is also indicated.*

### **Chapter 1–2–3: [Exploring and cleaning data]**

- **Status:** [ No Error Found]
- **Original Version:** [ N/A.]
- **Correct Version:** [N/A.]
- **Explanation:** [No errors were identified after checking the chapter.]

### **Chapter 4: [Feature Engineering and Encoding]**

- **Status:** [ No Error Found]
- **Original Version:** [ N/A.]
- **Correct Version:** [N/A.]
- **Explanation:** [No errors were identified after checking the chapter.]

### **Chapter 5: [Scaling and normalization]**

- **Status:** [ No Error Found]
- **Original Version:** [ N/A.]
- **Correct Version:** [N/A.]
- **Explanation:** [No errors were identified after checking the chapter.]

### Chapter 6: [Dealing with Outliers] 

- **Status:** Error Found / No Error Found
- **Explanation:**
One thing I noticed in this chapter is the difference between the Z-score result and the explanation given in the notebook. We talked about an outlier as a value that is far from the majority, and when I looked at the sample data, 100 is clearly far from the other values such as 10, 12, 15, 20, 21, and 22. However, when I ran the Z-score code, no outlier was detected because the Z-score of 100 was only 2.615, which is below the cutoff of 3.

At first, I thought that the code should be changed from > 3 to > 2.61 so that 100 would be detected. However, based on what was taught in the notebook, 3 is the standard cutoff for the Z-score method, so I decided not to change the code. The result is possible because the dataset only has 8 values, and the extreme value of 100 affects the mean and standard deviation, which makes its Z-score lower. This also shows why the IQR method can be more useful for this small sample, because it identifies 100 as an outlier without changing the Z-score cutoff.

### Chapter 7: [Feature Selection] 

- **Status:** Error Found
- **Original Version:** selector = RFECV(estimator, step=1, cv=5)
- **Correct Version:** selector = RFECV(estimator, step=1, cv=3)
- **Explanation:**
One error I noticed in this chapter is the use of cv=5 in the RFECV code. The dataset only has 7 samples, so using 5-fold cross-validation creates some validation sets with only one sample. This causes the warning UndefinedMetricWarning: R^2 score is not well-defined with less than two samples. To avoid this warning with this small dataset, cv=3 can be used instead of cv=5.

I also noticed the use of uppercase X and lowercase y in the Lasso section. At first, I thought that the uppercase X should be changed to lowercase x, but I learned that this is not an error. In machine learning, X is commonly used for the input features, while y is used for the target or output. Python is also case-sensitive, so X and x are treated as different variables.

Another thing I noticed is that the Filter Method output includes final grade with a correlation of 1.000000. Since final grade is the target that we want to predict, it should not be included as an input feature. The code is useful for showing the correlation, but when selecting actual features for prediction, final grade should be excluded.

### **Chapter 8: Constructing a Preprocessing Pipeline**

- **Status:** [Error Found / No Error Found]
- **Original Version:** [Incorrect code or statement / N/A]
- **Correct Version:** [Corrected code or statement / N/A]
- **Explanation:** [Explanation of the correction / No errors were identified after checking the chapter.]

### Chapter 9: [Real-World Application: Data Preprocessing]

- **Status:** Error Found 
- **Original Version:** # After discretization
plt.hist(titanic_preprocessed[:,2], alpha=0.5, label='After discretization')
plt.legend()
plt.show()
- **Correct Version:** # After discretization
plt.figure(figsize=(8, 5))
sns.countplot(
    x=data['Age'],
    order=['Child', 'Adult', 'Elderly']
)
plt.title("Age Distribution After Discretization")
plt.xlabel("Life Stage")
plt.ylabel("Count")
plt.show()
- **Explanation:**
One error I found is in Block 16, where titanic_preprocessed[:,2] was used to plot the age distribution after discretization. This is incorrect because column index 2 represents an encoded Embarked category, not the discretized Age column. Also, the age categories were created in data['Age'], while titanic_preprocessed was created before discretization. Therefore, the plot does not correctly show the age distribution after discretization.

---

## **Note on AI Tools**

# AI Notes of Pagar for Chapters 6–9

## Chapter 6: Outlier Detection and Treatment
I used an AI tool for this chapter because I was not very familiar with what strategy should be used after finding an outlier. I already understood how to find and identify an outlier, but I had limited knowledge about what to do with it afterward, so I asked AI for an explanation of the different strategies, such as Capping and Flooring and Removing Outliers. I also used AI to help me understand the difference between the Z-score and IQR results in this example

## Chapter 7: Feature Selection Using RFECV
I used an AI tool in this chapter mainly to help me identify and understand the error caused by cv=5 in the RFECV code. I also asked AI about the use of uppercase X and lowercase y because I initially thought that X should also be lowercase. I learned that this is a common machine learning convention where X represents the input features and y represents the target.

## Chapter 8:

(To be added)

## Chapter 9:
I used an AI tool to help me identify the possible errors in my code and understand what each result represents. After that, I analyzed the results myself and used my own understanding to answer the chapter questions. This helped me understand the purpose of each preprocessing step and how the visualizations represent the data.

---

## **References**

- McKinney, W. (2021). *Python for Data Analysis* (3rd ed.). O'Reilly.
- VanderPlas, J. *Python Data Science Handbook.*
- *Include any other websites, articles, or learning resources used in completing the case study.*
