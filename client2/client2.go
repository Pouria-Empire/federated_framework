package main

import (
	"context"
	"encoding/csv"
	"log"
	"os"
	"os/exec"
	"strconv"
	"time"

	pb "federated-learning-go/pb" // Adjust this path according to your Go module structure

	"google.golang.org/grpc"
)

// LoadWeights reads the model weights from the given CSV file
func LoadWeights(filename string) ([]float32, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.Read()
	if err != nil {
		return nil, err
	}

	// Convert the data to float32 slice
	weights := make([]float32, len(records))
	for i, v := range records {
		val, err := strconv.ParseFloat(v, 32)
		if err != nil {
			return nil, err
		}
		weights[i] = float32(val)
	}

	return weights, nil
}

func main() {
	// Connect to the server
	conn, err := grpc.Dial("localhost:50051", grpc.WithInsecure())
	if err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}
	defer conn.Close()

	client := pb.NewFederatedLearningClient(conn)

	// Execute the Python training script
	cmd := exec.Command("python", "train_tinyml.py") // Adjust the path as necessary
	cmd.Stdout = nil
	cmd.Stderr = nil
	if err := cmd.Run(); err != nil {
		log.Fatalf("Failed to execute training script: %v", err)
	}

	// Load model parameters from the CSV file after training
	modelParameters, err := LoadWeights("client_1_weights.csv") // Adjust filename as necessary for each client
	if err != nil {
		log.Fatalf("Failed to load model parameters: %v", err)
	}

	// Send model parameters to the server
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	response, err := client.UpdateModelParameters(ctx, &pb.ModelParameters{Parameters: modelParameters})
	if err != nil {
		log.Fatalf("Could not update model parameters: %v", err)
	}

	log.Printf("Server Response: %s", response.Message)
}
