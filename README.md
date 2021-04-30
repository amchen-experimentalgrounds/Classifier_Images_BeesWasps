These are project files for the subject Applied Data Science (MAST30034).
This group  project was done in Semester 2, 2020 at The University of Melbourne.

The task was inspired by real-world problems. 

The environment never exists under a state of equilibrium. 
Invasive species constantly threaten to devastate native populations. 
Weeds and pests multiply out-of-control and threaten agricultural lands.
Governments spend millions, if not billions, on protecting a nation's biosecurity.
Farmers can suffer devastating losses caused by pests.

Solutions exist, but they are old, ineffective and risky to human and environmental health.
For example, Glyphosate is one of the most used herbicides worldwide, and was classified as "probably carcinogenic in humans" by the World Health Organisation's International Agency for Research on Cancer.
In general, many chemical approaches have not been ideal for solving challenging issues, due to risks which can quickly spread. (See: DDT)

I proposed a different approach relying on AI and robotics.
This approach involves using computer vision to identify pests. This can be used alone as a surveillance tool.
A more ambitious solution would integrate this with automated elimination systems, which would have several advantages.
There are no likely side effects which occur beyond the immediate action of elimiating a pest (eg. substances spilling into waterways).
Pests cannot "evolve" to evade AI-enabled solutions. AI re-training is faster than resistance caused by evolution.

Our project did not address the elmination side in detail.

The task of identifying pests can be complex. The AI must be able to distinguish pests from non-pests. There are many different types of pests, and training a model can be difficult.
A solution which is fast to train and requires minimal data will be scalable, cost-efficient and thus ideal.

To achieve this, our model utilised a transfer learning approach (MobileNetV2), and expanded the number of training examples by augmenting the image data.
We trained this model to distinguish bees and wasps, which look similar.
Our model achieved a performance (measured by accuracy) of greater than 90%, which exceeded a 54% baseline.
Ultimately, we concluded that transfer-learning approaches are ideal for the problem of distinguishing pests and non-pests.

In the future, I intend to extend this project further by reducing the quality and availability of training data. This will be an important area to further study, as it will make the technology accessible to less developed regions.

AI + ROBOTICS IN THE FIELD
April 28, 2021: Vehicle uses 12 cameras and CO2 lasers to kill weeds: https://web.archive.org/web/20210430031736/https://www.intelligentliving.co/autonomous-robot-lasers-kill-weeds/
