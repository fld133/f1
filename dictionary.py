# dictionary.py


#  overall：整体场景 / 事件
OVERALL_KEYWORDS = []

#  action：可观察到的动作
ACTION_KEYWORDS = [
    "walking",
    "moving",
    "standing",
    "running",
    "drinking",
    "eating",
    "playing",
    "sitting",
    "swimming",
    "walking",
    "sitting",
    "reading",
    "driving",
    "writing"
]

#  object：实体 / 主体
OBJECT_KEYWORDS = [
    "elephant", "elephants",
    "animal",
    "clothes",
    "person",
    "people",
    "dog",
    "cat",
    "man","men",
    "woman","women",
    "child","children",
    "baby",
    "person",
    "car",
    "beer",
    "newspaper"

]

#  color：颜色属性
COLOR_KEYWORDS = [
    "black",
    "white",
    "red",
    "blue",
    "green",
    "yellow",
    "gray",
    "brown",
    "orange",
    "pink",
    "purple",
    "dark", "light",
]

#  count：数量 / 群体规模
COUNT_KEYWORDS = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "10 or more"
]

#  size：大小
SIZE_KEYWORDS = [
    "big", "large", "huge",
    "small", "tiny"
]

#  age：年龄阶段
AGE_KEYWORDS = [
    "baby", "young", "adult", "old","adult",
    "middle-aged",
    "child",
]

#  location：空间 / 场景位置
LOCATION_KEYWORDS = [
    "river",
    "forest",
    "grass",
    "field",
    "road",
    "home","house",
    "park",
    "swimming pool",
    "road",
    "gym",
]

#  统一给
TYPE_KEYWORDS_MAP = {
    "overall": OVERALL_KEYWORDS,
    "action": ACTION_KEYWORDS,
    "object": OBJECT_KEYWORDS,
    "color": COLOR_KEYWORDS,
    "count": COUNT_KEYWORDS,
    "size": SIZE_KEYWORDS,
    "age": AGE_KEYWORDS,
    "location": LOCATION_KEYWORDS,
}

