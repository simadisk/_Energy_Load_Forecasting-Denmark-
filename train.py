import mlflow
import random
import time

# 1. Σύνδεση με τον MLflow Server που τρέχει στο Docker μας
mlflow.set_tracking_uri("http://localhost:5000")

# 2. Ονομάζουμε το πείραμά μας
mlflow.set_experiment("AI_Geologist_Model_Training")

print("🚀 Ξεκινάει η εκπαίδευση του νέου μοντέλου...")

# Ξεκινάμε την καταγραφή του τρεχούμενου πειράματος
with mlflow.start_run(run_name="UNet3D_Test_Run"):
    
    # --- ΥΠΕΡΠΑΡΑΜΕΤΡΟΙ (Τις σώζουμε για να ξέρουμε τι ρυθμίσεις είχαμε) ---
    epochs = 10
    learning_rate = 0.001
    batch_size = 16
    
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("batch_size", batch_size)
    
    print(f"⚙️ Ρυθμίσεις: Epochs={epochs}, LR={learning_rate}")
    
    # --- ΕΚΠΑΙΔΕΥΣΗ (Εδώ μπαίνει ο κανονικός PyTorch κώδικας) ---
    for epoch in range(epochs):
        time.sleep(1) # Κάνουμε μια μικρή παύση για να προσομοιώσουμε τον χρόνο εκπαίδευσης
        
        # Υπολογισμός ενός εικονικού Loss που μειώνεται και Accuracy που αυξάνεται
        loss = 2.0 / (epoch + 1) + random.uniform(0, 0.1)
        accuracy = 100 - (loss * 30)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.2f}%")
        
        # --- ΤΟ ΜΥΣΤΙΚΟ: Στέλνουμε τις μετρικές στο MLflow σε πραγματικό χρόνο ---
        mlflow.log_metric("train_loss", loss, step=epoch)
        mlflow.log_metric("val_accuracy", accuracy, step=epoch)
        
    print("✅ Η εκπαίδευση ολοκληρώθηκε με επιτυχία!")