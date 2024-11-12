# Step 1: Install Additional Dependencies for MongoDB and CORS
npm install mongoose cors

# Step 2: Update src/server.js to Include MongoDB Connection, CORS, and API Routes
cat <<EOL > src/server.js
import express from 'express';
import mongoose from 'mongoose';
import dotenv from 'dotenv';
import cors from 'cors';

dotenv.config();
const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// MongoDB Connection
mongoose.connect(process.env.MONGO_URI, { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => console.log("MongoDB connected"))
  .catch(err => console.log(err));

// Define Schema and Model
const ItemSchema = new mongoose.Schema({ name: String });
const Item = mongoose.model('Item', ItemSchema);

// Routes
app.get('/api/items', async (req, res) => {
  const items = await Item.find();
  res.json(items);
});

app.post('/api/items', async (req, res) => {
  const newItem = new Item({ name: req.body.name });
  await newItem.save();
  res.json(newItem);
});

app.listen(PORT, () => {
  console.log(\`Server running on http://127.0.0.1:\${PORT}\`);
});
EOL

# Step 3: Update Environment Variables in .env for MongoDB URI
echo "MONGO_URI=mongodb+srv://<username>:<password>@cluster.mongodb.net/<dbname>?retryWrites=true&w=majority" >> .env

# Step 4: Install Frontend (React) and Set Up Basic Project
npx create-react-app client
cd client
npm install axios

# Step 5: Add Proxy for API Requests in client/package.json
sed -i '' '/"scripts": {/a\
"proxy": "http://127.0.0.1:3000",
' package.json

# Step 6: Create Basic API Interaction in client/src/App.js
cat <<EOL > src/App.js
import React, { useEffect, useState } from 'react';
import axios from 'axios';

function App() {
  const [items, setItems] = useState([]);
  const [newItem, setNewItem] = useState('');

  useEffect(() => {
    axios.get('/api/items').then(response => setItems(response.data));
  }, []);

  const addItem = () => {
    axios.post('/api/items', { name: newItem }).then(response => {
      setItems([...items, response.data]);
      setNewItem('');
    });
  };

  return (
    <div>
      <h1>Item List</h1>
      <ul>
        {items.map(item => <li key={item._id}>{item.name}</li>)}
      </ul>
      <input value={newItem} onChange={(e) => setNewItem(e.target.value)} placeholder="New Item" />
      <button onClick={addItem}>Add Item</button>
    </div>
  );
}

export default App;
EOL

# Step 7: Start Both Server and Client for Local Testing

# In one terminal, start the server
cd ../
node src/server.js

# In another terminal, start the React client
cd client
npm start

# Step 8: Prepare for Deployment to Heroku (for server) and Netlify (for frontend)

# Step 8.1: Deploy Backend to Heroku
# Initialize Git Repository (if not already done)
git init
heroku create
echo "web: node src/server.js" > Procfile
git add .
git commit -m "Deploy backend to Heroku"
git push heroku master

# Step 8.2: Deploy Frontend to Netlify
# In the "client" directory, build the React app
npm run build
# Go to https://app.netlify.com/ and connect your GitHub repository to deploy
