import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      title: 'Flutter + FastAPI',
      home: HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  String status = "No status fetched";
  Map<String, dynamic> config = {
    "data_path": "",
    "model_size": "small",
    "image_size": 224,
    "num_classes": 2,
    "epochs": 1,
    "batch_size": 32,
    "learning_rate": 0.001,
    "output_path": "./output",
  };

  // Fetch backend status
  Future<void> fetchStatus() async {
    final response = await http.get(Uri.parse('http://127.0.0.1:8000/status'));
    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      setState(() {
        status = data["status"];
      });
    } else {
      setState(() {
        status = "Failed to fetch status";
      });
    }
  }

  // Update backend config
  Future<void> updateConfig() async {
    final response = await http.post(
      Uri.parse('http://127.0.0.1:8000/config'),
      headers: {"Content-Type": "application/json"},
      body: json.encode(config),
    );
    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(data["message"])),
      );
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Failed to update config")),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Flutter + FastAPI')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text("Backend Status: $status"),
            const SizedBox(height: 10),
            ElevatedButton(
              onPressed: fetchStatus,
              child: const Text('Fetch Status'),
            ),
            const SizedBox(height: 20),
            const Text("Update Config"),
            TextField(
              decoration: const InputDecoration(labelText: "Data Path"),
              onChanged: (value) {
                config["data_path"] = value;
              },
            ),
            DropdownButton<String>(
              value: config["model_size"],
              items: ["small", "medium", "large"]
                  .map((size) => DropdownMenuItem(
                        value: size,
                        child: Text(size),
                      ))
                  .toList(),
              onChanged: (value) {
                setState(() {
                  config["model_size"] = value!;
                });
              },
            ),
            ElevatedButton(
              onPressed: updateConfig,
              child: const Text('Update Config'),
            ),
          ],
        ),
      ),
    );
  }
}
