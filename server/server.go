package main

import (
	"context"
	"log"
	"net"
	"sync"

	pb "federated-learning-go/pb" // Adjust this path according to your Go module structure

	"google.golang.org/grpc"
)

// Server represents the gRPC server
type server struct {
	pb.UnimplementedFederatedLearningServer
	modelWeights [][]float32 // To hold model weights from clients
	mu           sync.Mutex   // Mutex to manage concurrent access to modelWeights
	clientsCount int          // Count of connected clients
}

// UpdateModelParameters receives model parameters from clients
func (s *server) UpdateModelParameters(ctx context.Context, req *pb.ModelParameters) (*pb.Response, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Store received model parameters
	s.modelWeights = append(s.modelWeights, req.Parameters)
	s.clientsCount++

	log.Printf("Received model parameters from client %d: %v", s.clientsCount, req.Parameters)

	// If we have received parameters from 3 clients, notify them
	if s.clientsCount == 3 { // Adjust this number based on your actual client count
		s.notifyClients()
	}

	return &pb.Response{Message: "Model parameters received successfully"}, nil
}

// Notify clients that all parameters have been received
func (s *server) notifyClients() {
	log.Println("All client parameters received. Notifying clients...")
	// Notify logic can be implemented here (e.g., sending a gRPC message back to clients)
	// This is just a placeholder for now.
}

// Start the server
func startServer() {
	listener, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}
	grpcServer := grpc.NewServer()
	pb.RegisterFederatedLearningServer(grpcServer, &server{}) // Register the server

	log.Println("Starting gRPC server on port 50051...")
	if err := grpcServer.Serve(listener); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}

func main() {
	startServer()
}
