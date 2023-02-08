using UnityEngine;

public class DetectEmotion : MonoBehaviour
{
    public FetchEmotion ex;
    public GameObject targetGameObject;
    public string emotion;
    void Start()
    {
        if (targetGameObject != null)
        {
            ex = targetGameObject.GetComponent<FetchEmotion>();
        }
        InvokeRepeating("NewEmotion", 0, 8f);
    }

    void NewEmotion()
    {
            whatEmotion();
            Debug.Log(emotion);
            if (ex != null)
            {
                ex.ntrl = 0;
                ex.sad = 0;
                ex.happy = 0;
            }
    }

    void whatEmotion()
    {
        int largest = ex.ntrl;
        emotion = "Neutral";
        if (ex.happy > largest)
        {
            largest = ex.happy;
            emotion = "happy";
        }
        if (ex.sad > largest)
        {
            largest = ex.sad;
            emotion = "sad";
        }
    }
}
