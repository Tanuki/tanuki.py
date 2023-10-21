import React, { useState, useEffect } from "react";
import axios from "axios";

function App() {
  const [input, setInput] = useState("");
  const [todoItems, setTodoItems] = useState([]);

  useEffect(() => {
    const storedItems = JSON.parse(localStorage.getItem("todoItems") || "[]");
    setTodoItems(storedItems);
  }, []);

  useEffect(() => {
    localStorage.setItem("todoItems", JSON.stringify(todoItems));
  }, [todoItems]);

  const addTodoItem = async () => {
    try {
      const response = await axios.post("http://localhost:8000/create_todolist_items/", { input });
      const newItems = response.data;
      setTodoItems([...todoItems, ...newItems]);
    } catch (error) {
      console.error("An error occurred while fetching data: ", error);
    }
  };

  const deleteTodoItem = (index) => {
    const newItems = todoItems.slice();
    newItems.splice(index, 1);
    setTodoItems(newItems);
  };

  return (
    <div className="App">
      <h1>To-Do List</h1>
      <div>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
        <button onClick={addTodoItem}>Add</button>
      </div>
      <ul>
        {todoItems.map((item, index) => (
          <li key={index}>
            {item.goal} - {item.deadline} - {item.people.join(", ")}
            <button onClick={() => deleteTodoItem(index)}>Delete</button>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default App;
