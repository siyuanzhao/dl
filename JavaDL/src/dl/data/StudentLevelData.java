package dl.data;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

public class StudentLevelData {

	public int num; //total number of problems the student answers
	public ArrayList<Integer> problemSets = new ArrayList<>(); // the sequence of problem sets
	public ArrayList<Integer> correctness = new ArrayList<>(); // the sequence of student's correctness
	
	public static void main(String[] args) {
		Set<Integer> set = new HashSet<>();
		set.add(1);
		System.out.println(DataPreprocess.getIndex(set, 1));
		set.add(2);
		System.out.println(DataPreprocess.getIndex(set, 2));
		set.add(2);
		System.out.println(DataPreprocess.getIndex(set, 2));
		set.add(3);
		System.out.println(DataPreprocess.getIndex(set, 2));
		set.add(8);
		System.out.println(DataPreprocess.getIndex(set, 2));
		set.add(7);
		System.out.println(DataPreprocess.getIndex(set, 2));
		set.add(6);
		System.out.println(DataPreprocess.getIndex(set, 2));
	}
}
