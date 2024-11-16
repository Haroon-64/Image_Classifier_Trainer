import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class ApiService {
   String baseUrl = "http://127.0.0.1:8000";

   

  Future<void> configure(String dataPath, String modelSize, int epochs, int batchSize) async {
    // enter url from user input
    final response = await http.post(
      Uri.parse("$baseUrl/configure"),
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({
        "data_path": dataPath,
        "model_size": modelSize,
        "epochs": epochs,
        "batch_size": batchSize,
      }),
    );
    if (response.statusCode == 200) {
      if (kDebugMode) {
        print("Configuration successful");
      }
    } else {
      throw Exception("Failed to configure");
    }
  }

  Future<void> trainModel() async {
    final response = await http.post(Uri.parse("$baseUrl/train"));
    if (response.statusCode == 200) {
      if (kDebugMode) {
        print("Training started");
      }
    } else {
      throw Exception("Failed to start training");
    }
  }

  Future<String> getStatus() async {
    final response = await http.get(Uri.parse("$baseUrl/status"));
    if (response.statusCode == 200) {
      return jsonDecode(response.body)["status"];
    } else {
      throw Exception("Failed to fetch status");
    }
  }
}
