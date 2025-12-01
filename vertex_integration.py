class GeminiVertexAIProcessor:
    """Main processor class for Gemini Vertex AI"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.client = self._initialize_client()
        self.loader_factory = DataLoaderFactory(self.config)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config file: {str(e)}")
            raise
    
    def _initialize_client(self) -> genai.Client:
        """Initialize Gemini Vertex AI client"""
        try:
            # Set environment variables for Vertex AI
            os.environ["GOOGLE_CLOUD_PROJECT"] = self.config['gemini']['project_id']
            os.environ["GOOGLE_CLOUD_LOCATION"] = self.config['gemini']['location']
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
            
            return genai.Client()
        except Exception as e:
            logger.error(f"Error initializing Gemini client: {str(e)}")
            raise
    
    def process_file(self, file_path: str, question: str = None, prompt_type: str = "default") -> Dict[str, Any]:
        """Process a file with Gemini Vertex AI"""
        try:
            # Load data using appropriate loader
            loader = self.loader_factory.get_loader(file_path)
            processed_data = loader.load(file_path)
            
            # Generate appropriate prompt
            prompt = self._generate_prompt(processed_data.data_type, question, prompt_type)
            
            # Process with Gemini
            response = self._send_to_gemini(processed_data, prompt)
            
            # Format and return results
            return self._format_results(processed_data, response, prompt)
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise
    
    def _generate_prompt(self, data_type: str, question: str = None, prompt_type: str = "default") -> str:
        """Generate appropriate prompt based on data type and user question"""
        if question:
            return question
        
        prompts_config = self.config['prompts']
        
        if data_type == 'image':
            return prompts_config['image_analysis'].get(prompt_type, prompts_config['image_analysis']['default'])
        elif data_type == 'document':
            return prompts_config['document_summary'].get(prompt_type, prompts_config['document_summary']['default'])
        elif data_type == 'spreadsheet':
            return prompts_config['spreadsheet_analysis'].get(prompt_type, prompts_config['spreadsheet_analysis']['default'])
        else:
            return prompts_config['text_processing'].get(prompt_type, prompts_config['text_processing']['default'])
    
    def _send_to_gemini(self, processed_data: ProcessedData, prompt: str) -> Any:
        """Send data and prompt to Gemini Vertex AI"""
        try:
            if processed_data.data_type == 'image':
                # Handle image content
                response = self.client.models.generate_content(
                    model=self.config['gemini']['model'],
                    contents=[
                        types.Part.from_bytes(
                            data=processed_data.content,
                            mime_type=self._get_mime_type(processed_data.source_file)
                        ),
                        prompt
                    ],
                    config=types.GenerateContentConfig(
                        max_output_tokens=self.config['gemini']['max_tokens'],
                        temperature=self.config['gemini']['temperature']
                    )
                )
            elif processed_data.data_type == 'spreadsheet':
                # Handle spreadsheet data
                spreadsheet_text = self._spreadsheet_to_text(processed_data.content)
                response = self.client.models.generate_content(
                    model=self.config['gemini']['model'],
                    contents=[spreadsheet_text, prompt],
                    config=types.GenerateContentConfig(
                        max_output_tokens=self.config['gemini']['max_tokens'],
                        temperature=self.config['gemini']['temperature']
                    )
                )
            else:
                # Handle text content
                response = self.client.models.generate_content(
                    model=self.config['gemini']['model'],
                    contents=[processed_data.content, prompt],
                    config=types.GenerateContentConfig(
                        max_output_tokens=self.config['gemini']['max_tokens'],
                        temperature=self.config['gemini']['temperature']
                    )
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error sending to Gemini: {str(e)}")
            raise
    
    def _get_mime_type(self, file_path: str) -> str:
        """Get MIME type for file"""
        extension = Path(file_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp'
        }
        return mime_types.get(extension, 'application/octet-stream')
    
    def _spreadsheet_to_text(self, data_dict: Dict[str, Any]) -> str:
        """Convert spreadsheet data to text format for Gemini"""
        lines = []
        lines.append("Spreadsheet Data:")
        lines.append(f"Columns: {', '.join(data_dict['columns'])}")
        lines.append(f"Total Rows: {data_dict['shape'][0]}")
        lines.append("\nFirst 10 rows of data:")
        
        for i, row in enumerate(data_dict['data'][:10]):
            row_text = f"Row {i+1}: "
            row_text += ', '.join([f"{col}: {row[col]}" for col in data_dict['columns']])
            lines.append(row_text)
        
        if data_dict['shape'][0] > 10:
            lines.append(f"... and {data_dict['shape'][0] - 10} more rows")
        
        return '\n'.join(lines)
    
    def _format_results(self, processed_data: ProcessedData, response: Any, prompt: str) -> Dict[str, Any]:
        """Format results for output"""
        results = {
            'source_file': processed_data.source_file,
            'data_type': processed_data.data_type,
            'prompt': prompt,
            'response': response.text,
            'metadata': {
                **processed_data.metadata,
                'gemini_model': self.config['gemini']['model'],
                'processing_timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        # Save results if configured
        if self.config['output']['save_results']:
            self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save results to file"""
        output_dir = Path(self.config['output']['output_directory'])
        output_dir.mkdir(exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.json"
        
        output_path = output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {output_path}")
