import 'package:flutter/material.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  // ignore: library_private_types_in_public_api
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final TextEditingController urlController = TextEditingController();
  String status = "Idle";

  // Function to validate if the entered text is a valid URL
  bool isValidUrl(String url) {
    final uri = Uri.tryParse(url);
    return uri != null && (uri.isScheme('http') || uri.isScheme('https'));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Image Classifier")),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            TextField(
              controller: urlController,
              decoration: InputDecoration(
                labelText: "Enter API URL",
                hintText: "https://example.com",
                errorText: isValidUrl(urlController.text) || urlController.text.isEmpty
                    ? null
                    : "Please enter a valid URL",
              ),
              keyboardType: TextInputType.url,
              onChanged: (text) {
                setState(() {});  // Rebuild to update error text if needed
              },
            ),
            const SizedBox(height: 20),
            // Example button to use the URL input
            ElevatedButton(
              onPressed: () {
                if (isValidUrl(urlController.text)) {
                  // Use the URL for API calls or configuration
                  print("URL Entered: ${urlController.text}");
                  setState(() {
                    status = "URL configured: ${urlController.text}";
                  });
                } else {
                  print("Invalid URL entered.");
                }
              },
              child: const Text("Set URL"),
            ),
            const SizedBox(height: 20),
            Text("Status: $status"),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    urlController.dispose();
    super.dispose();
  }
}
