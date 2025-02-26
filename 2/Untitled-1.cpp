#include <iostream>
#include <vector>

using namespace std;

// Définition de la classe Order
class Order {
public:
    int orderID;
    double price;
    double quantity;
    
    Order(int id, double p, double q) : orderID(id), price(p), quantity(q) {}
};

// Fonction pour exécuter la stratégie iceberg
vector<Order> icebergStrategy(double totalQuantity, double icebergQuantity, double price) {
    vector<Order> orders;
    double remainingQuantity = totalQuantity;
    
    while (remainingQuantity > 0) {
        double currentQuantity = min(icebergQuantity, remainingQuantity);
        orders.push_back(Order(orders.size() + 1, price, currentQuantity));
        remainingQuantity -= currentQuantity;
    }
    
    return orders;
}

int main() {
    // Paramètres de la stratégie iceberg
    double totalQuantity = 1000.0;
    double icebergQuantity = 100.0;
    double price = 50.0;
    
    // Exécuter la stratégie iceberg
    vector<Order> icebergOrders = icebergStrategy(totalQuantity, icebergQuantity, price);
    
    // Afficher les ordres générés
    cout << "Orders generated by iceberg strategy:" << endl;
    for (const auto& order : icebergOrders) {
        cout << "Order ID: " << order.orderID << ", Price: " << order.price << ", Quantity: " << order.quantity << endl;
    }
    
    return 0;
}
