library(shiny)
library(ggplot2)
library(dplyr)
library(DT)
library(caret)
library(pROC)
library(reshape2)

# Load datasets
student_mat <- read.csv("student-mat.csv")
student_por <- read.csv("student-por.csv")

# Combine datasets
student_data <- bind_rows(
  mutate(student_mat, Subject = "Math"),
  mutate(student_por, Subject = "Portuguese")
)

# Define UI
ui <- fluidPage(
  titlePanel("Student Performance Dashboard with Advanced Analysis"),
  
  sidebarLayout(
    sidebarPanel(
      selectInput("subject", "Select Subject", choices = c("All", "Math", "Portuguese"), selected = "All"),
      uiOutput("dynamicFilters"),
      actionButton("update", "Update View")
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Summary Table", DTOutput("summaryTable")),
        tabPanel("Demographic Overview", 
                 plotOutput("genderPlot"),
                 plotOutput("ageDistributionPlot")),
        tabPanel("Parental Insights", plotOutput("parentEducationPlot")),
        tabPanel("Study Habits", plotOutput("studyTimePlot")),
        tabPanel("Marks Insights", 
                 plotOutput("marksDistributionPlot"),
                 plotOutput("marksVsStudytimePlot"),
                 plotOutput("marksVsFailuresPlot")),
        tabPanel("Correlation Matrix", plotOutput("correlationMatrixPlot")),
        tabPanel("ROC Curves", plotOutput("rocPlot"))
      )
    )
  )
)

server <- function(input, output, session) {
  # Generate dynamic filter inputs
  output$dynamicFilters <- renderUI({
    data <- if (input$subject == "All") student_data else student_data %>% filter(Subject == input$subject)
    
    filter_inputs <- lapply(names(data), function(col) {
      if (is.numeric(data[[col]])) {
        sliderInput(
          inputId = paste0("filter_", col),
          label = paste("Filter by", col),
          min = min(data[[col]], na.rm = TRUE),
          max = max(data[[col]], na.rm = TRUE),
          value = range(data[[col]], na.rm = TRUE)
        )
      } else {
        selectInput(
          inputId = paste0("filter_", col),
          label = paste("Filter by", col),
          choices = unique(data[[col]]),
          selected = unique(data[[col]]),
          multiple = TRUE
        )
      }
    })
    do.call(tagList, filter_inputs)
  })
  
  # Filter data based on user inputs
  filtered_data <- reactive({
    data <- if (input$subject == "All") student_data else student_data %>% filter(Subject == input$subject)
    
    for (col in names(data)) {
      filter_id <- paste0("filter_", col)
      if (!is.null(input[[filter_id]])) {
        if (is.numeric(data[[col]])) {
          data <- data %>% filter(data[[col]] >= input[[filter_id]][1] & data[[col]] <= input[[filter_id]][2])
        } else {
          data <- data %>% filter(data[[col]] %in% input[[filter_id]])
        }
      }
    }
    data
  })
  
  # Render summary table with horizontal scrolling
  output$summaryTable <- renderDT({
    datatable(
      filtered_data(),
      options = list(
        scrollX = TRUE,  # Enable horizontal scrolling
        pageLength = 10  # Set default number of rows displayed
      ),
      class = "display nowrap"  # Ensure proper styling for scrolling
    )
  })
  
  # Gender Distribution
  output$genderPlot <- renderPlot({
    data <- filtered_data()
    ggplot(data, aes(x = "", fill = sex)) +
      geom_bar(width = 1) +
      coord_polar("y") +
      labs(title = "Gender Distribution", fill = "Gender") +
      theme_minimal()
  })
  
  # Age Distribution
  output$ageDistributionPlot <- renderPlot({
    data <- filtered_data()
    ggplot(data, aes(x = age)) +
      geom_histogram(binwidth = 1, fill = "skyblue", color = "black") +
      labs(title = "Age Distribution", x = "Age", y = "Count") +
      theme_minimal()
  })
  
  # Parental Education Levels
  output$parentEducationPlot <- renderPlot({
    data <- filtered_data()
    ggplot(data, aes(x = Medu, fill = factor(Medu))) +
      geom_bar() +
      labs(title = "Parental Education Levels (Mother)", x = "Education Level", y = "Count", fill = "Level") +
      theme_minimal()
  })
  
  # Study Time vs. Failures
  output$studyTimePlot <- renderPlot({
    data <- filtered_data()
    ggplot(data, aes(x = studytime, y = failures)) +
      geom_jitter(alpha = 0.6, color = "blue") +
      labs(title = "Study Time vs Failures", x = "Study Time (hours)", y = "Number of Failures") +
      theme_minimal()
  })
  
  # Marks Distribution
  output$marksDistributionPlot <- renderPlot({
    data <- filtered_data()
    ggplot(data, aes(x = G3)) +
      geom_histogram(binwidth = 1, fill = "lightgreen", color = "black") +
      labs(title = "Marks Distribution", x = "Final Marks (G3)", y = "Frequency") +
      theme_minimal()
  })
  
  # Marks vs Study Time
  output$marksVsStudytimePlot <- renderPlot({
    data <- filtered_data()
    ggplot(data, aes(x = studytime, y = G3)) +
      geom_point(color = "purple", alpha = 0.6) +
      geom_smooth(method = "lm", color = "red", se = FALSE) +
      labs(title = "Marks vs Study Time", x = "Study Time (hours)", y = "Final Marks (G3)") +
      theme_minimal()
  })
  
  # Marks vs Failures
  output$marksVsFailuresPlot <- renderPlot({
    data <- filtered_data()
    ggplot(data, aes(x = failures, y = G3)) +
      geom_jitter(alpha = 0.6, color = "orange") +
      labs(title = "Marks vs Failures", x = "Number of Failures", y = "Final Marks (G3)") +
      theme_minimal()
  })
  
  # Correlation Matrix
  output$correlationMatrixPlot <- renderPlot({
    data <- filtered_data()
    num_data <- data %>% select_if(is.numeric)
    corr_matrix <- cor(num_data, use = "complete.obs")
    melted_corr <- melt(corr_matrix)
    
    ggplot(melted_corr, aes(x = Var1, y = Var2, fill = value)) +
      geom_tile() +
      scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1, 1)) +
      labs(title = "Correlation Matrix", x = "", y = "") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  })
  
  # ROC Curves with Accuracy Calculation and Display
  output$rocPlot <- renderPlot({
    data <- filtered_data()
    
    # Define binary target variable for modeling
    data$pass_fail <- ifelse(data$G3 >= 10, "Pass", "Fail")
    data$pass_fail <- factor(data$pass_fail, levels = c("Fail", "Pass"))
    
    # Split data into training and testing sets
    set.seed(123)
    train_idx <- createDataPartition(data$pass_fail, p = 0.7, list = FALSE)
    train_data <- data[train_idx, ]
    test_data <- data[-train_idx, ]
    
    # Train models
    knn_model <- train(pass_fail ~ studytime + failures + age, data = train_data, method = "knn")
    svm_model <- train(pass_fail ~ studytime + failures + age, data = train_data, method = "svmLinear",
                       trControl = trainControl(classProbs = TRUE))
    gbm_model <- train(pass_fail ~ studytime + failures + age, data = train_data, method = "gbm", verbose = FALSE)
    
    # Generate predictions
    knn_pred <- predict(knn_model, test_data, type = "prob")[, 2]
    svm_pred <- predict(svm_model, test_data, type = "prob")[, 2]
    gbm_pred <- predict(gbm_model, test_data, type = "prob")[, 2]
    
    # Remove NA values from predictions
    valid_idx <- complete.cases(knn_pred, svm_pred, gbm_pred)
    knn_pred <- knn_pred[valid_idx]
    svm_pred <- svm_pred[valid_idx]
    gbm_pred <- gbm_pred[valid_idx]
    test_data <- test_data[valid_idx, ]
    
    # Generate ROC curves
    knn_roc <- roc(test_data$pass_fail, knn_pred)
    svm_roc <- roc(test_data$pass_fail, svm_pred)
    gbm_roc <- roc(test_data$pass_fail, gbm_pred)
    
    # Plot ROC curves and accuracy
    plot(knn_roc, col = "blue", main = "ROC Curves for KNN, SVM, GBM")
    lines(svm_roc, col = "red")
    lines(gbm_roc, col = "green")
    legend("bottomright", legend = c(
      paste("KNN (AUC =", round(auc(knn_roc), 2), ")"),
      paste("SVM (AUC =", round(auc(svm_roc), 2), ")"),
      paste("GBM (AUC =", round(auc(gbm_roc), 2), ")")
    ), col = c("blue", "red", "green"), lty = 1)
  })
}

# Run the app
shinyApp(ui = ui, server = server)
