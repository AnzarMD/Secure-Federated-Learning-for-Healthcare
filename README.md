# Secure-Federated-Learning-for-Healthcare
This project explores privacy-preserving machine learning for healthcare IoT data by implementing and comparing two different training approaches: a centralized model and a federated learning model.

In traditional centralized systems, data from multiple devices or organizations is collected and stored in one location to train a model. While effective, this approach can raise serious privacy and security concerns, especially when dealing with sensitive healthcare information.

To address this challenge, this project also implements a federated learning approach, where models are trained collaboratively across multiple devices or data sources without moving the raw data to a central server. Each participant trains the model locally and only shares model updates, helping protect sensitive patient information.

To further strengthen privacy, the project integrates differential privacy techniques, which add an additional layer of protection to ensure that individual data points cannot be identified from the trained model.

By comparing centralized training and federated training with differential privacy, this project demonstrates how AI models can still achieve strong predictive performance while maintaining strict privacy standards.

Key Goals

Explore secure AI training methods for healthcare data

Compare centralized and federated learning approaches

Implement differential privacy to protect sensitive information

Evaluate how privacy-preserving methods impact model performance and scalability

Motivation

Healthcare data is highly sensitive and often distributed across many devices, hospitals, and organizations. Moving all data to a single location is not always practical or safe. This project demonstrates how collaborative AI systems can be built while keeping data private, which is essential for real-world healthcare applications.
