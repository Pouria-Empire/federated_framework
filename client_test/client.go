package main

import (
	"context"
	"log"
	"math/rand"
	"time"

	pb "federated-learning-go/pb"

	"google.golang.org/grpc"
)

func generateRandomModelParams() []float32 {
	// Simulate random model parameters
	rand.Seed(time.Now().UnixNano())
	params := make([]float32, 10) // Let's assume we have 10 parameters
	for i := range params {
		params[i] = rand.Float32() // Random float between 0 and 1
	}
	return params
}

func main() {
	// Connect to the server.
	conn, err := grpc.Dial("localhost:50051", grpc.WithInsecure())
	if err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}
	defer conn.Close()

	client := pb.NewFederatedLearningClient(conn)

	// Generate local model parameters.
	localParams := generateRandomModelParams()
	log.Println("Generated local model parameters:", localParams)

	// Send model parameters to the server.
	_, err = client.SendModelParameters(context.Background(), &pb.ModelParameters{Weights: localParams})
	if err != nil {
		log.Fatalf("Failed to send model parameters: %v", err)
	}
	log.Println("Model parameters sent to server.")

	// Request global model parameters from the server.
	globalParams, err := client.GetGlobalParameters(context.Background(), &pb.Empty{})
	if err != nil {
		log.Fatalf("Failed to get global parameters: %v", err)
	}
	log.Println("Received global model parameters from server:", globalParams.Weights)
}
