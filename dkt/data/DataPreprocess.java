package dl.data;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Set;

public class DataPreprocess {

	public static void main(String[] args) {
		
		Set<Integer> set = new HashSet<>();
		String csvFile = "/home/siyuan/Downloads/problem_logs_2015.csv";
		BufferedReader br = null;
		String line = "";
		String cvsSplitBy = ",";
		int preStuId = 0;
		List<StudentLevelData> list = null;
		try {
			StudentLevelData sld = null;
			list = new ArrayList<>();
			br = new BufferedReader(new FileReader(csvFile));
			while ((line = br.readLine()) != null) {
				// use comma as separator
				String[] columns = line.split(cvsSplitBy);
				int studentId = Integer.valueOf(columns[0]);
				int problemSetId = Integer.valueOf(columns[2]);
				set.add(problemSetId);
				float floatCorrectness = Float.valueOf(columns[3]);
				int correctness = 0;
				if(floatCorrectness == 1) {
					correctness = 1;
				}
				if(preStuId == studentId) { // data for the same student
					sld.num += 1;
					sld.problemSets.add(getIndex(set, problemSetId));
					sld.correctness.add(correctness);
					
				} else { // data from different student
					if(sld != null) {
						list.add(sld);
					}
					sld = new StudentLevelData();
					sld.num = 1;
					sld.problemSets.add(getIndex(set, problemSetId));
					sld.correctness.add(correctness);
					preStuId = studentId;
				}

			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (br != null) {
				try {
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		// shuffle the list
		long seed = System.nanoTime();
		Collections.shuffle(list, new Random(seed));
		
		int totalSize = list.size();
		int trainSize = totalSize*8/10;
		
		List<StudentLevelData> trainList = list.subList(0, trainSize);
		List<StudentLevelData> testList = list.subList(trainSize, list.size());
		Iterator<StudentLevelData> iter = trainList.iterator();
		// write to files
		try {
			FileWriter writer = new FileWriter("/home/siyuan/programs/eclipse/workspace/DL/train.csv");
			while(iter.hasNext()) {
				StudentLevelData item = iter.next();
				writer.append(item.num + "\n");
				writer.append(item.problemSets.toString().replace("[", "").replace("]", "")
	            .replace(", ", ","));
				writer.append("\n");
				writer.append(item.correctness.toString().replace("[", "").replace("]", "")
			            .replace(", ", ","));
				writer.append("\n");
				
			}
			writer.flush();
			writer.close();
		} catch(IOException e) {
			e.printStackTrace();
		}
		
		iter = testList.iterator();
		try {
			FileWriter writer = new FileWriter("/home/siyuan/programs/eclipse/workspace/DL/test.csv");
			while(iter.hasNext()) {
				StudentLevelData item = iter.next();
				writer.append(item.num + "\n");
				writer.append(item.problemSets.toString().replace("[", "").replace("]", "")
	            .replace(", ", ","));
				writer.append("\n");
				writer.append(item.correctness.toString().replace("[", "").replace("]", "")
			            .replace(", ", ","));
				writer.append("\n");
				
			}
			writer.flush();
			writer.close();
		} catch(IOException e) {
			e.printStackTrace();
		}
	}
	
	public static int getIndex(Set<? extends Object> set, Object value) {
		   int result = 0;
		   for (Object entry:set) {
		     if (entry.equals(value)) return result;
		     result++;
		   }
		   return -1;
		 }
}
