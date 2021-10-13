setwd("C:/Users/vsgui/Documents/RProjects/DSA-ML-Project02")
getwd()


#Installing the packages
install.packages("Amelia") #Functions to work with null values
install.packages("caret") #Allows to build ML models
install.packages("ggplot2") #Build Graphics
install.packages("dplyr") #To manipulate data
install.packages("reshape") #Modify the shape of data
install.packages("randomForest") #ML
install.packages("e1071") #ML


#Loading the library
library(Amelia)
library(ggplot2)
library(caret)
library(reshape)
library(randomForest)
library(dplyr)
library(e1071)

#loading the data
#data used from ("https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients")
dados_clientes <- read.csv("dados/dataset.csv")

#Visualize the data
View(dados_clientes)
dim(dados_clientes)
str(dados_clientes)
summary(dados_clientes)

#Cleaning the data

#Removing the column ID
dados_clientes$ID <- NULL
dim(dados_clientes)
View(dados_clientes)

#Rename the column Class (The Target)
colnames(dados_clientes)
colnames(dados_clientes)[24] <- "Inadimplente"
View(dados_clientes)

#Verify if there are no N/A values
sapply(dados_clientes, function(x) sum(is.na(x))) #Creates a loop over the data and applies a function during the execution
?missmap
missmap(dados_clientes, main = "Valores Missing Observados")#Graphic visualization - Proof that we don't have N/A values
dados_clientes <- na.omit(dados_clientes) #If there are N/A values we are going to "omit"

#Converting the attributes gender, education, age, marital status to categories (Currently integer to Factor(char in C))
colnames(dados_clientes)
colnames(dados_clientes)[2] <- "Genero"
colnames(dados_clientes)[3] <- "Escolaridade"
colnames(dados_clientes)[4] <- "Estado_Civil"
colnames(dados_clientes)[5] <- "Idade"
colnames(dados_clientes)
View(dados_clientes)

#Gender
View(dados_clientes$Genero)
str(dados_clientes$Genero)
summary((dados_clientes$Genero))
?cut #Convert numeric to Factor
dados_clientes$Genero <- cut(dados_clientes$Genero,
                             c(0,1,2),
                             labels= c("Masculino",
                                       "Feminino"))
View(dados_clientes$Genero)
str(dados_clientes$Genero)
summary((dados_clientes$Genero))

#Education First edit(Letting the N/A data to see what happen in the future with the model)
str(dados_clientes$Escolaridade)
summary((dados_clientes$Escolaridade))
dados_clientes$Escolaridade <- cut(dados_clientes$Escolaridade,
                             c(0,1,2,3,4),
                             labels= c("Pos Graduado",
                                       "Graduado",
                                       "Ensino Medio",
                                       "Outros"))
View(dados_clientes$Escolaridade)
str(dados_clientes$Escolaridade)
summary((dados_clientes$Escolaridade))

#Marital Status
str(dados_clientes$Estado_Civil)
summary((dados_clientes$Estado_Civil))
dados_clientes$Estado_Civil <- cut(dados_clientes$Estado_Civil,
                                   c(-1,0,1,2,3),
                                   labels= c("Desconhecido",
                                             "Casado",
                                             "Solteiro",
                                             "Outro"))
View(dados_clientes$Estado_Civil)
str(dados_clientes$Estado_Civil)
summary((dados_clientes$Estado_Civil))

#Age
str(dados_clientes$Idade)
summary((dados_clientes$Idade))
hist(dados_clientes$Idade) #Creates a histogram for the Data
dados_clientes$Idade <- cut(dados_clientes$Idade, #Working with the data divided in groups
                                   c(0,30,50,100),
                                   labels= c("Jovem",
                                             "Adulto",
                                             "Idoso"))
                                             
View(dados_clientes$Idade)
str(dados_clientes$Idade)
summary((dados_clientes$Idade))

View(dados_clientes)

#Converting the variables related to payments
#We are going to use as.factor because we are not changing the data just the type
dados_clientes$PAY_0 <- as.factor(dados_clientes$PAY_0)
dados_clientes$PAY_2 <- as.factor(dados_clientes$PAY_2)
dados_clientes$PAY_3 <- as.factor(dados_clientes$PAY_3)
dados_clientes$PAY_4 <- as.factor(dados_clientes$PAY_4)
dados_clientes$PAY_5 <- as.factor(dados_clientes$PAY_5)
dados_clientes$PAY_6 <- as.factor(dados_clientes$PAY_6)

#Checking the data
str(dados_clientes)
sapply(dados_clientes, function(x) sum(is.na(x))) #We can now see N/A values
missmap(dados_clientes, main = "Valores Missing Observados")
dados_clientes <- na.omit(dados_clientes) #We opted to remove the N/A values since we do not know what to use instead
missmap(dados_clientes, main = "Valores Missing Observados")
dim(dados_clientes) #We can see that we removed 345 values


#Last visualization
View(dados_clientes)

#Last change is to make sure that our Target value is also set to right type
dados_clientes$Inandimplente <- as.factor(dados_clientes$Inandimplente)
str(dados_clientes$Inandimplente)

#Total of people that default vs non default
table(dados_clientes$Inandimplente)

#The percentage for the table above
prop.table(table(dados_clientes$Inandimplente))

#The data show that more people did not default, and that's normal since more people usually pay on time
#But in order to train the model we will have to make sure we give the same amount of data for the algorithm
#Otherwise it will learn much more about a group than and that's not what we want

#Plotting the distribution with ggplot2
qplot(Inandimplente, data = dados_clientes, geom="bar") +
  theme(axis.text.x =  element_text(angle = 90, hjust = 1))

#Set seed is used to randomly divide the data to sample and test
set.seed(12345)

#Select the rows based on the variable of people that either default or not
?createDataPartition
#75% of the data will go to training 
indice <- createDataPartition(dados_clientes$Inandimplente, p = 0.75, list = FALSE)
dim(indice)

#Create the "Dataset" for the training
dados_treino <- dados_clientes[indice,] #I want to select that data from dads_clientes and import all columns
dim(dados_treino)
table(dados_treino$Inandimplente)
prop.table(table(dados_treino$Inandimplente))

#Comparing the percentage for the data in the training Dataset and the original Dataset
compara_dados <- cbind(prop.table(table(dados_treino$Inandimplente)),
                       prop.table(table(dados_clientes$Inandimplente)))
colnames(compara_dados) <- c("Treinamento", "Original")
compara_dados

#Now we are going to see the same information but in the form of graphic 
#We use the Melt function to convert columns to lines
?reshape2 ::melt
melt_compara_dados <- melt(compara_dados)
melt_compara_dados

ggplot(melt_compara_dados, aes(x =X1, y= value)) +
  geom_bar(aes(fill = X2), stat = "identity", position = "dodge") +
             theme(axis.text.x = element_text(angle = 90, hjust = 1))

#Now we create a Dataset for testing the model later
#For that we are going to give all but the data given to the Training Dataset
dados_teste <- dados_clientes[-indice,] #For that all we do is add - in front of the data we want to exclude
dim(dados_teste)
dim(dados_treino)

#Machine Learn model - Begin
#First Model- Random Forest model (Series of Tree algorithms inside a single model)
?randomForest
modelo_v1 <- randomForest(Inandimplente ~ ., data = dados_treino)
modelo_v1

#Evaluating the model
plot(modelo_v1)

#Previsions with this model
previsoes_v1 <- predict(modelo_v1, dados_teste)

#Using the confusion Matrix - ("Evaluates the model")
?caret :: confusionMatrix
cm_v1 <- caret::confusionMatrix(previsoes_v1, dados_teste$Inandimplente, positive = "1")
cm_v1

#Calculating the precision, recall e F1-Score, all used to evaluate the model
y <- dados_teste$Inandimplente
y_pred_v1 <- previsoes_v1

precision <- posPredValue(y_pred_v1, y)
precision

recall <- sensitivity(y_pred_v1, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

#The previous model was made with the data not balanced... now we are going to train it with balanced data
#We are going to use the Technic of over sampling with SMOTE
install.packages("devtools")
library(devtools)
install_local('C:/Users/vsgui/Documents/RProjects/DSA-ML-Project02/dados/DMwR_0.4.1.tar.gz')
library(DMwR)
?SMOTE #Algorithm - Synthetic Minority Over-sampling Technique

table(dados_treino$Inandimplente)
prop.table(table(dados_treino$Inandimplente))
set.seed(9560)
dados_treino_bal <- SMOTE(Inandimplente ~ ., data = dados_treino)

#Second model, now with balanced data
modelo_v2 <- randomForest(Inandimplente ~ ., data = dados_treino_bal)
modelo_v2

plot(modelo_v2)

previsoes_v2 <- predict(modelo_v2, dados_teste)

cm_v2 <- caret::confusionMatrix(previsoes_v2, dados_teste$Inandimplente, positive = "1")
cm_v2

y <- dados_teste$Inandimplente
y_pred_v2 <- previsoes_v2

precision <- posPredValue(y_pred_v2, y)
precision

recall <- sensitivity(y_pred_v2, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

#Importance of the variables
View(dados_treino_bal)
varImpPlot(modelo_v2)

# Visualizing the most important variables
imp_var <- varImpPlot(modelo_v2)
varImportance <- data.frame(Variables = row.names(imp_var),
                            Importance = round(imp_var[ ,'MeanDecreaseGini'],2))

#Creating a rank for the variables
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#', dense_rank(desc(Importance))))

#Plotting the graphic with ggplot2
ggplot(rankImportance,
       aes( x = reorder(Variables, Importance),
            y = Importance,
            fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust = 0,
            vjust = 0.55,
            size = 4,
            colour = 'red') +
  labs(x = 'Variables')
coord_flip()

#Third model - Now only using the most relevant variables
colnames(dados_treino_bal)
modelo_v3 <- randomForest(Inandimplente ~ PAY_0 + PAY_2 + PAY_3 + PAY_AMT1 + PAY_AMT2 + PAY_5 + BILL_AMT1,
                          dados = dados_treino_bal)
modelo_v3

plot(modelo_v3)

previsoes_v3 <- predict(modelo_v3, dados_teste)

cm_v3 <- caret::confusionMatrix(previsoes_v3, dados_teste$Inandimplente, positive = "1")
cm_v3

y <- dados_teste$Inandimplente
y_pred_v3 <- previsoes_v3

precision <- posPredValue(y_pred_v3, y)
precision

recall <- sensitivity(y_pred_v3, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

#Saving the model in the disk
saveRDS(modelo_v3, file = "modelo/modelo_v3.rsd")

#Loading a model later
modelo_final <- readRDS("modelo/modelo_v3.rds")

#We can plot latter with power BI or Shine for example

#Last training giving new data
# function c in R creates a vector
PAY_0 <- c(0, 0, 0)
PAY_2 <- c(0, 0, 0)
PAY_3 <- c(1, 0, 0)
PAY_AMT1 <- c(1100, 1000, 1200)
PAY_AMT2 <- c(1500, 1300, 1150)
PAY_5 <- c(0, 0, 0)
BILL_AMT1(350, 420, 280)

#Concatenate in a DataFrame
novos_clientes <- data.frame(PAY_0, PAY_2, PAY_3, PAY_AMT1, PAY_AMT2, PAY_5, BILL_AMT1)
View(novos_clientes)

#Previsions for the new data
previsoes_novos_clientes <- predict(modelo_final, novos_clientes) #It won't work-- this is just for testing

#Checking the type of data 
str(dados_treino_bal)
str(novos_clientes)

#Converting the new data
novos_clientes$PAY_0 <- factor(novos_clientes$PAY_0, levels = (dados_treino_bal$PAY_0))
novos_clientes$PAY_2 <- factor(novos_clientes$PAY_2, levels = (dados_treino_bal$PAY_2))
novos_clientes$PAY_3 <- factor(novos_clientes$PAY_3, levels = (dados_treino_bal$PAY_3))
novos_clientes$PAY_5 <- factor(novos_clientes$PAY_5, levels = (dados_treino_bal$PAY_5))
str(novos_clientes)

#And now we can predict for the new clients
previsoes_novos_clientes <- predict(modelo_final, novos_clientes)
View(previsoes_novos_clientes)

#---------------The END ------------------------------#

