class Student:
    
    
    def __init__(self, name, score):
        self.name = name
        self.score = score
        
    
    @staticmethod
    def top_scorer(students):
        highest = students[0]
        for student in students:
            if student.score > highest.score:
                highest = student
                
        return highest
    
    @staticmethod
    def display_above_fifty(students):
        for student in students:
            if student.score > 50:
                print(student.name, student.score, "\n")

            
students = []


for i in range(3):
    name  = input(f"enter name of student {i+1}: ")
    score = float(input(f"Enter score for {name}: "))
    students.append(Student(name, score))
    
print("\n\n above 50 marks students: ")
Student.display_above_fifty(students)


print("highest score: ")
highest  = Student.top_scorer(students)
print(highest.name, " " ,highest.score)