import org.tensorflow.*;

public class Demo1 {

	public static void main(String[] args) throws Exception {

		try (Graph g1 = new Graph();) {

			// adding the name C1 and value 2 to the graph g1
			Output<Integer> c1 = addConstant(g1, "C1", 2);
			Output<Integer> c2 = addConstant(g1, "C2", 12);

			// adding components of graph together
			Output<Operation> a = addAddOperation(g1, c1, c2);

			try (Session s = new Session(g1);
				Tensor output = s.runner().fetch(a).run().get(0)) {
				System.out.println(output.intValue());
			}
		}
	}
}