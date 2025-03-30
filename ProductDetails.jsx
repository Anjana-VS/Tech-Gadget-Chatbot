import React from 'react';

function ProductDetails({ items }) {
  if (!items || items.length === 0) return null;

  return (
    <div className="product-details">
      <h3>Recommended Gadgets:</h3>
      <ul>
        {items.map((item, index) => (
          <li key={index}>
            <strong>{item["Product Name"]}</strong> - {item["Features"]} <br />
            Category: {item["Category"]}, Brand: {item["Brand"]}, Price: ${item["Price"]} <br />
            Specifications: {item["Specifications"]} <br />
            Rating: {item["Popularity Score"]}, User Review: {item["User Reviews"]}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default ProductDetails;